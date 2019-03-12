/*
 Copyright 2016-2018 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include <cstring>
#include <mpi.h>
#include <stdlib.h>

#include "comm.hpp"
#include "common.hpp"

#ifdef INTERNAL_ENV_UPDATE
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#define START_COMM(reqExpr, name, leftIdx, rightIdx)        \
  do {                                                      \
      size_t reqIdx;                                        \
      for (reqIdx = leftIdx; reqIdx < rightIdx; reqIdx++)   \
      {                                                     \
          MLSL_LOG(DEBUG, "start %s: reqIdx %zu, "          \
                   "elems %zu, offset_in_bytes %zu",        \
                   name, reqIdx, reqCounts[reqIdx],         \
                   reqOffsets[reqIdx]);                     \
          reqExpr;                                          \
      }                                                     \
  } while (0)

#define WAIT_COMM(count, requests, statuses)           \
  do {                                                 \
      MLSL_LOG(DEBUG, "waitall: count %zu, req[0] %d", \
               count, requests[0]);                    \
      MPI_Waitall(count, requests, statuses);          \
      size_t idx;                                      \
      for (idx = 0; idx < count; idx++)                \
      {                                                \
          requests[idx] = MPI_REQUEST_NULL;            \
      }                                                \
  } while (0)

#define TEST_COMM(count, requests, flag, statuses)     \
  do {                                                 \
      MLSL_LOG(DEBUG, "testall: count %zu, req[0] %d", \
               count, requests[0]);                    \
      MPI_Testall(count, requests, &flag, statuses);   \
      if (flag)                                        \
      {                                                \
          size_t idx;                                  \
          for (idx = 0; idx < count; idx++)            \
          {                                            \
              requests[idx] = MPI_REQUEST_NULL;        \
          }                                            \
      }                                                \
  } while (0)

#define MLSL_NUM_SERVERS_ENV        "MLSL_NUM_SERVERS"
#define MLSL_SERVER_AFFINITY_ENV    "MLSL_SERVER_AFFINITY"
#define MLSL_MAX_SHORT_MSG_SIZE_ENV "MLSL_MAX_SHORT_MSG_SIZE"
#define MLSL_THP_THRESHOLD_MB_ENV   "MLSL_THP_THRESHOLD_MB"
#define MLSL_ALLTOALL_SPLIT_ENV     "MLSL_ALLTOALL_SPLIT"
#define MLSL_USE_MPI_FORCE_ENV      "MLSL_USE_MPI_FORCE"

#define STR_OR_NULL(str) ((str) ? str : "null")

#define EXPECTED_MPI_VERSION "Intel(R) MPI Library 2019"

#define COMM_KEY                     "ep_idx"
#define THREAD_COUNT_MAX             16
#define THREAD_COUNT_DEFAULT         4
#define MAX_SHORT_MSG_SIZE           1024
#define THP_THRESHOLD_MB_DEFAULT     128
#define ALLTOALL_SPLIT_PARTS_DEFAULT 1
#define USE_MPI_FORCE_DEFAULT        0

#define ONE_MB 1048576
#define TWO_MB 2097152

#define GATHER_TAG  20000
#define SCATTER_TAG 30000

#define EP_IDX_MAX_LEN       (4)
#define THREAD_COUNT_MAX_LEN (4)

namespace MLSL
{
    ProcessGroupImpl* globalPg = NULL;
    ProcessGroupImpl* selfPg = NULL;
    size_t myIdx;
    size_t mySize;
    size_t threadCount = THREAD_COUNT_DEFAULT;
    size_t commCount = THREAD_COUNT_DEFAULT;
    size_t maxShortMsgSize = MAX_SHORT_MSG_SIZE;
    size_t thpThresholdMb = THP_THRESHOLD_MB_DEFAULT;
    size_t allToAllSplitParts = ALLTOALL_SPLIT_PARTS_DEFAULT;
    int useMpiForce = USE_MPI_FORCE_DEFAULT;
    int isExternalInit = 0;

    class ProcessGroupImpl
    {
    private:
        ProcessGroup* bp;
        MPI_Comm* epComms;
        ProcessGroupImpl(const ProcessGroupImpl& processGroup);
        ProcessGroupImpl& operator=(const ProcessGroupImpl& processGroup);

    public:
        ProcessGroupImpl(MPI_Comm comm)
        {
            size_t idx = 0, size = 0;
            MPI_Comm_rank(comm, (int*)&idx);
            MPI_Comm_size(comm, (int*)&size);
            bp = new ProcessGroup(this, idx, size);
            epComms = (MPI_Comm*)MLSL_MALLOC(commCount * sizeof(MPI_Comm), CACHELINE_SIZE);

            MLSL_LOG(DEBUG, "ProcessGroupImpl: idx %zu, size %zu", idx, size);
            MPI_Info info;
            MPI_Info_create(&info);
            for (idx = 0; idx < commCount; idx++)
            {
                MPI_Comm_dup(comm, &epComms[idx]);
                char epIdxStr[1024] = { 0 };
                sprintf(epIdxStr, "%zu", idx);
                MPI_Info_set(info, COMM_KEY, epIdxStr);
                MPI_Comm_set_info(epComms[idx], info);
            }
            MPI_Info_free(&info);
        }

        ~ProcessGroupImpl()
        {
            if (bp)
            {
                delete bp;
                bp = NULL;
            }

            size_t idx;
            for (idx = 0; idx < commCount; idx++)
            {
                MPI_Comm_free(&epComms[idx]);
            }
            MLSL_FREE(epComms);
        }

        MPI_Comm GetComm() { return epComms[0]; }
        MPI_Comm* GetEpComms() { return epComms; }
        ProcessGroup* GetBackP() { return bp; }
    };

    class CommRequestImpl
    {
    private:
        size_t epCount;
        bool isNullReq;
        CommRequest* bp;
        MPI_Datatype dataType;
        size_t dataTypeSize;
        MPI_Op redOp;
        MPI_Comm comm;
        MPI_Comm* epComms;
        size_t length;
        CommDesc::CompType compType;
        CommOp::ReqType reqType;
        MPI_Request* nonBlockReqs;
        size_t nonBlockReqCount;
        size_t rootIdx;
        size_t myRank;
        size_t numprocs;
        size_t groupSize;

        size_t* reqCounts;
        size_t* reqOffsets;

        /* for alltoallv */
        int* sndCounts;
        int* rcvCounts;
        int* sndOffsets;
        int* rcvOffsets;

        void SetMPIType(DataType dType)
        {
            if (dType == DT_DOUBLE)     dataType = MPI_DOUBLE;
            else if (dType == DT_FLOAT) dataType = MPI_FLOAT;
            else if (dType == DT_BYTE)  dataType = MPI_CHAR;
            else if (dType == -1)       dataType = MPI_CHAR;
            else MLSL_ASSERT(0, "unsupported datatype %d", dType);
            dataTypeSize = 0;
            MPI_Type_size(dataType, (int*)&dataTypeSize);
        }

        void SetMPIOp(CommOp::ReduceOp rOp)
        {
            if (rOp == CommOp::RO_SUM)
                redOp = MPI_SUM;
            else if (rOp == CommOp::RO_MIN)
                redOp = MPI_MIN;
            else if (rOp == CommOp::RO_MAX)
                redOp = MPI_MAX;
            else
                MLSL_ASSERT(0, "unsupported reduction operation %d", rOp);
        }

        CommRequestImpl(const CommRequestImpl& op);
        CommRequestImpl& operator=(const CommRequestImpl& op);

    public:
        CommRequestImpl(DataType dataType, int opUniqueId, CommDesc::CompType compType)
                        : epCount(1),
                          isNullReq(false),
                          redOp(MPI_OP_NULL),
                          comm(MPI_COMM_NULL),
                          epComms(NULL),
                          length(0),
                          compType(CommDesc::NONE),
                          reqType(CommOp::None),
                          nonBlockReqs(NULL),
                          nonBlockReqCount(0),
                          rootIdx(0),
                          myRank(0),
                          numprocs(0),
                          groupSize(0),
                          reqCounts(NULL),
                          reqOffsets(NULL),
                          sndCounts(NULL),
                          rcvCounts(NULL),
                          sndOffsets(NULL),
                          rcvOffsets(NULL)
        {
            bp = new CommRequest(this, dataType, opUniqueId, compType);
            SetMPIType(dataType);
        }

        ~CommRequestImpl()
        {
            if (nonBlockReqs)
            {
                MLSL_FREE(nonBlockReqs);
                nonBlockReqs = NULL;
            }

            if (bp)
            {
                delete bp;
                bp = NULL;
            }

            if (reqCounts)
            {
                MLSL_FREE(reqCounts);
                reqCounts = NULL;
            }

            if (reqOffsets)
            {
                MLSL_FREE(reqOffsets);
                reqOffsets = NULL;
            }

            if (sndCounts)
            {
                MLSL_FREE(sndCounts);
                sndCounts = NULL;
            }

            if (rcvCounts)
            {
                MLSL_FREE(rcvCounts);
                rcvCounts = NULL;
            }

            if (sndOffsets)
            {
                MLSL_FREE(sndOffsets);
                sndOffsets = NULL;
            }

            if (rcvOffsets)
            {
                MLSL_FREE(rcvOffsets);
                rcvOffsets = NULL;
            }

        }

        CommRequest* GetBackP() { return bp; }

        void Setup()
        {
            size_t reqIdx;
            CommDesc* cd = bp->GetDesc();
            if (cd->GetOpCount() == 0)
            {
                isNullReq = true;
                return;
            }

            isNullReq = false;
            MLSL_ASSERT(cd->GetOpCount() == 1, "GetOpCount > 1 is not supported yet, GetOpCount %zu", cd->GetOpCount());
            CommOp* op = cd->GetOp(0);
            compType = cd->GetComputeType();
            comm = op->GetProcessGroup()->GetImpl()->GetComm();
            epComms = op->GetProcessGroup()->GetImpl()->GetEpComms();
            reqType = op->GetReqType();
            groupSize = op->GetProcessGroup()->GetSize();

            MPI_Comm_rank(comm, (int*)&myRank);
            MPI_Comm_size(comm, (int*)&numprocs);

            if (reqType == CommOp::AllGather)
            {
                CommOpAllGather* aOp = static_cast<CommOpAllGather*>(op);
                length = aOp->GetLen();
                bp->tmpSz = length * op->GetProcessGroup()->GetSize() * dataTypeSize;
                nonBlockReqCount = op->GetProcessGroup()->GetSize();
            }
            else if (reqType == CommOp::AllGatherv)
            {
                CommOpAllGatherv* aOp = static_cast<CommOpAllGatherv*>(op);
                length = aOp->GetLen();

                size_t* rcounts = aOp->GetRecvCounts();
                size_t bufSize = rcounts[0];
                for (size_t i = 1; i < op->GetProcessGroup()->GetSize(); i++)
                     bufSize += rcounts[i];
                bp->tmpSz = bufSize * dataTypeSize;
                nonBlockReqCount = op->GetProcessGroup()->GetSize();
            }
            else if (reqType == CommOp::ReduceScatter)
            {
                CommOpReduceScatter* aOp = static_cast<CommOpReduceScatter*>(op);
                length = aOp->GetLen();
                SetMPIOp(aOp->GetOp());

                /* In-place not supported when messages split across endpoints */
                bp->isInPlace = false;

                if (bp->isInPlace)
                    bp->tmpSz = length * op->GetProcessGroup()->GetSize() * dataTypeSize;
                else
                    bp->tmpSz = length * (op->GetProcessGroup()->GetSize() + 1) * dataTypeSize;

                nonBlockReqCount = op->GetProcessGroup()->GetSize();
            }
            else if (reqType == CommOp::AllReduce)
            {
                CommOpAllReduce* aOp = static_cast<CommOpAllReduce*>(op);
                SetMPIOp(aOp->GetOp());
                length = aOp->GetLen();
                bp->tmpSz = length * dataTypeSize;
            }
            else if (reqType == CommOp::Reduce)
            {
                CommOpReduce* aOp = static_cast<CommOpReduce*>(op);
                SetMPIOp(aOp->GetOp());
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                bp->tmpSz = length * dataTypeSize;
            }
            else if (reqType == CommOp::Gather)
            {
                CommOpGather* aOp = static_cast<CommOpGather*>(op);
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                bp->isInPlace = false;
                nonBlockReqCount = op->GetProcessGroup()->GetSize() + 1;
            }
            else if (reqType == CommOp::Scatter)
            {
                CommOpScatter* aOp = static_cast<CommOpScatter*>(op);
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                nonBlockReqCount = op->GetProcessGroup()->GetSize() + 1;
            }
            else if (reqType == CommOp::Bcast)
            {
                CommOpBcast* aOp = static_cast<CommOpBcast*>(op);
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                bp->tmpSz = length * dataTypeSize;
            }
            else if (reqType == CommOp::AlltoAll)
            {
                CommOpAlltoAll* aOp = static_cast<CommOpAlltoAll*>(op);
                length = aOp->GetLen();

                /* In-place not supported when messages split across endpoints */
                bp->isInPlace = false;

                if (bp->isInPlace)
                    bp->tmpSz = length * op->GetProcessGroup()->GetSize() * dataTypeSize;
                else
                    bp->tmpSz = length * (op->GetProcessGroup()->GetSize() * 2) * dataTypeSize;

                /* Two requests per process - alltoall using send/recv */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() * 2;
            }
            else if (reqType == CommOp::AlltoAllv)
            {
                CommOpAlltoAllv* aOp = static_cast<CommOpAlltoAllv*>(op);
                size_t* rcounts = aOp->GetRecvCounts();
                for (size_t i = 0; i < numprocs; i++)
                  length += rcounts[i];

                sndCounts = (int*)MLSL_MALLOC(sizeof(int) * numprocs, CACHELINE_SIZE);
                rcvCounts = (int*)MLSL_MALLOC(sizeof(int) * numprocs, CACHELINE_SIZE);
                sndOffsets = (int*)MLSL_MALLOC(sizeof(int) * numprocs, CACHELINE_SIZE);
                rcvOffsets = (int*)MLSL_MALLOC(sizeof(int) * numprocs, CACHELINE_SIZE);
                MLSL_ASSERT(sndCounts && rcvCounts && sndOffsets && rcvOffsets, "Can't allocate memory");

                for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                {
                    sndCounts[procIdx] = aOp->GetSendCounts()[procIdx];
                    rcvCounts[procIdx] = aOp->GetRecvCounts()[procIdx];
                    sndOffsets[procIdx] = aOp->GetSendOffsets()[procIdx];
                    rcvOffsets[procIdx] = aOp->GetRecvOffsets()[procIdx];
                }

                /* In-place not supported when messages split across endpoints */
                bp->isInPlace = false;
                bp->tmpSz = length * dataTypeSize;

                /* Two requests per process - alltoallv using send/recv */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() * 2;
            }
            else if (reqType == CommOp::Barrier)
            {
                length = 0;
            }
            else
                MLSL_ASSERT(0, "reqType %d is not supported yet", reqType);

            epCount = commCount;
            if (length <= maxShortMsgSize  && reqType != CommOp::AllGatherv) epCount = 1;
            if (length < epCount) epCount = 1;

            if (reqType == CommOp::AllGather  ||
                reqType == CommOp::AllGatherv ||
                reqType == CommOp::ReduceScatter)
            {
                reqCounts = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                reqOffsets = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);

                size_t* recvCounts = NULL;
                if (reqType == CommOp::AllGatherv)
                {
                    CommOpAllGatherv* aOp = static_cast<CommOpAllGatherv*>(op);
                    recvCounts = aOp->GetRecvCounts();
                }

                for (reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                {
                    if (reqType == CommOp::AllGatherv)
                        reqCounts[reqIdx] = recvCounts[reqIdx];
                    else
                        reqCounts[reqIdx] = length;

                    if (reqIdx == 0)
                        reqOffsets[reqIdx] = 0;
                    else
                        reqOffsets[reqIdx] = reqOffsets[reqIdx - 1] + reqCounts[reqIdx - 1] * dataTypeSize;
                }
            }
            else if (reqType == CommOp::Gather ||
                     reqType == CommOp::Scatter)
            {
                reqCounts = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                reqOffsets = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                MLSL_ASSERT(reqCounts && reqOffsets, "NULL pointers");
                for (reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                {
                    reqCounts[reqIdx] = length;

                    if (reqIdx == (nonBlockReqCount - 1))
                        reqOffsets[reqIdx] = 0;
                    else
                        reqOffsets[reqIdx] = length * dataTypeSize * reqIdx;
                }
            }
            else if (reqType == CommOp::AlltoAll ||
                     reqType == CommOp::AlltoAllv)
            {
                nonBlockReqCount *= allToAllSplitParts;
                reqCounts = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                reqOffsets = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                MLSL_ASSERT(reqCounts && reqOffsets, "NULL pointers");

                size_t elemCount, elemOffset;
                for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                {
                    size_t recvRank = (myRank - procIdx + numprocs) % numprocs;
                    if (reqType == CommOp::AlltoAllv)
                    {
                        elemCount = rcvCounts[recvRank];
                        elemOffset = rcvOffsets[recvRank];
                    }
                    else
                    {
                        elemCount = length;
                        elemOffset = recvRank * length;
                    }
                    for (size_t splitIdx = 0; splitIdx < allToAllSplitParts; splitIdx++)
                    {
                        size_t globalSplitIdx = procIdx * allToAllSplitParts + splitIdx;
                        reqCounts[globalSplitIdx] = elemCount / allToAllSplitParts;
                        if (splitIdx == (allToAllSplitParts - 1))
                            reqCounts[globalSplitIdx] += elemCount % allToAllSplitParts;
                        reqOffsets[globalSplitIdx] =
                            (elemOffset + (splitIdx * (elemCount / allToAllSplitParts))) * dataTypeSize;
                    }
                }
                size_t idxOffset = numprocs * allToAllSplitParts;
                for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                {
                    size_t sendRank = (myRank + procIdx + numprocs) % numprocs;
                    if (reqType == CommOp::AlltoAllv)
                    {
                        elemCount = sndCounts[sendRank];
                        elemOffset = sndOffsets[sendRank];
                    }
                    else
                    {
                        elemCount = length;
                        elemOffset = sendRank * length;
                    }
                    for (size_t splitIdx = 0; splitIdx < allToAllSplitParts; splitIdx++)
                    {
                        size_t globalSplitIdx = idxOffset + procIdx * allToAllSplitParts + splitIdx;
                        reqCounts[globalSplitIdx] = elemCount / allToAllSplitParts;
                        if (splitIdx == (allToAllSplitParts - 1))
                            reqCounts[globalSplitIdx] += elemCount % allToAllSplitParts;
                        reqOffsets[globalSplitIdx] =
                            (elemOffset + (splitIdx * (elemCount / allToAllSplitParts))) * dataTypeSize;
                    }
                }
            }
            else if (reqType == CommOp::Bcast     ||
                     reqType == CommOp::AllReduce ||
                     reqType == CommOp::Reduce)
            {
                /* pure data-parallel approach */
                nonBlockReqCount = epCount;
                reqCounts = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                reqOffsets = (size_t*)MLSL_MALLOC(nonBlockReqCount * sizeof(size_t), CACHELINE_SIZE);
                size_t baseCount = length / nonBlockReqCount;
                size_t reqIdx;
                for (reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                {
                    reqCounts[reqIdx] = baseCount;
                    reqOffsets[reqIdx] = reqIdx * reqCounts[reqIdx] * dataTypeSize;
                }
                reqCounts[nonBlockReqCount - 1] += length % nonBlockReqCount;
            }

            if (reqType != CommOp::Barrier)
                MLSL_ASSERT(nonBlockReqCount, "nonBlockReqCount");

            nonBlockReqs = (MPI_Request*)MLSL_MALLOC(nonBlockReqCount * sizeof(MPI_Request), CACHELINE_SIZE);
            MLSL_LOG(DEBUG, "nonBlockReqs %p, nonBlockReqCount %zu", nonBlockReqs, nonBlockReqCount);
            MLSL_ASSERT(nonBlockReqs, "nonBlockReqs is NULL, nonBlockReqCount %zu", nonBlockReqCount);
            for (size_t idx = 0; idx < nonBlockReqCount; idx++)
                nonBlockReqs[idx] = MPI_REQUEST_NULL;

            MLSL_LOG(DEBUG, "(%p): op %s, threadCount %zu, commCount %zu, epCount %zu, reqCount %zu, count %zu",
                     bp, bp->GetDesc()->GetOp(0)->GetReqName().c_str(),
                     threadCount, commCount, epCount, nonBlockReqCount, length);

            for (size_t idx = 0; idx < nonBlockReqCount; idx++)
            {
                MLSL_LOG(DEBUG, "(%p): op %s, req %zu, count %zu, byte_offset %zu",
                         bp, bp->GetDesc()->GetOp(0)->GetReqName().c_str(), idx,
                         reqCounts ? reqCounts[idx] : 0,
                         reqOffsets ? reqOffsets[idx] : 0);
            }
        }

        int Start(void* buf, void* retBuf)
        {
            if (isNullReq) return 0;

            if (reqType == CommOp::AllGather ||
                reqType == CommOp::AllGatherv)
            {
                if (buf == retBuf)
                {
                    START_COMM(
                        MPI_Ibcast((char*)buf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType,
                                   reqIdx, epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "ibcast", 0, nonBlockReqCount
                    );
                }
                else
                {
                    memcpy((char*)retBuf + reqOffsets[myRank], buf, reqCounts[myRank] * dataTypeSize);
                    START_COMM(
                        MPI_Ibcast((char*)retBuf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType,
                                   reqIdx, epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "ibcast", 0, nonBlockReqCount
                    );
                }
            }
            else if (reqType == CommOp::Bcast)
            {
                START_COMM(
                    MPI_Ibcast((char*)buf + reqOffsets[reqIdx],
                               reqCounts[reqIdx], dataType, rootIdx,
                               epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                    "ibcast", 0, nonBlockReqCount
                );
            }
            else if (reqType == CommOp::ReduceScatter)
            {
                if (buf == retBuf)
                {
                    MPI_Reduce_scatter_block(MPI_IN_PLACE, buf, length,
                                             dataType, redOp, epComms[0]);                    
                }
                else
                {
                    START_COMM(
                        MPI_Ireduce((char*)buf + reqOffsets[reqIdx],
                                    (char*)retBuf,
                                    reqCounts[reqIdx], dataType, redOp, reqIdx,
                                    epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "ireduce", 0, nonBlockReqCount
                    );
                }
            }
            else if (reqType == CommOp::AllReduce)
            {
                if (buf == retBuf)
                {
                    START_COMM(
                        MPI_Iallreduce(MPI_IN_PLACE,
                                       (char*)buf + reqOffsets[reqIdx],
                                       reqCounts[reqIdx], dataType, redOp,
                                       epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "iallreduce", 0, nonBlockReqCount
                    );
                }
                else
                {
                    START_COMM(
                        MPI_Iallreduce((char*)buf + reqOffsets[reqIdx],
                                       (char*)retBuf + reqOffsets[reqIdx],
                                       reqCounts[reqIdx], dataType, redOp,
                                       epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "iallreduce", 0, nonBlockReqCount
                    );
                }
            }
            else if (reqType == CommOp::Reduce)
            {
                if (buf == retBuf)
                {
                    if (myRank == rootIdx)
                    {
                        START_COMM(
                            MPI_Ireduce(MPI_IN_PLACE,
                                        (char*)buf + reqOffsets[reqIdx],
                                        reqCounts[reqIdx], dataType, redOp, rootIdx,
                                        epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                            "ireduce", 0, nonBlockReqCount
                        );
                    }
                    else
                    {
                        START_COMM(
                            MPI_Ireduce((char*)buf + reqOffsets[reqIdx],
                                        (char*)buf + reqOffsets[reqIdx],
                                        reqCounts[reqIdx], dataType, redOp, rootIdx,
                                        epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                            "ireduce", 0, nonBlockReqCount
                        );
                    }
                }
                else
                {
                    START_COMM(
                        MPI_Ireduce((char*)buf + reqOffsets[reqIdx],
                                    (char*)retBuf + reqOffsets[reqIdx],
                                    reqCounts[reqIdx], dataType, redOp, rootIdx,
                                    epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "ireduce", 0, nonBlockReqCount
                    );
                }
            }
            else if (reqType == CommOp::Gather)
            {
                if (myRank == rootIdx)
                {
                    START_COMM(
                        MPI_Irecv((char*)retBuf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType,
                                  reqIdx, GATHER_TAG, epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "irecv", 0, (nonBlockReqCount - 1)
                    );
                }
                START_COMM(
                    MPI_Isend((char*)buf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType, rootIdx,
                              GATHER_TAG, epComms[myRank % epCount], &nonBlockReqs[reqIdx]),
                    "isend", (nonBlockReqCount - 1), nonBlockReqCount
                );
            }
            else if (reqType == CommOp::Scatter)
            {
                if (myRank == rootIdx)
                {
                    START_COMM(
                        MPI_Isend((char*)buf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType,
                                  reqIdx, SCATTER_TAG, epComms[reqIdx % epCount], &nonBlockReqs[reqIdx]),
                        "isend", 0, (nonBlockReqCount - 1)
                    );
                }
                START_COMM(
                    MPI_Irecv((char*)retBuf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType, rootIdx,
                               SCATTER_TAG, epComms[myRank % epCount], &nonBlockReqs[reqIdx]),
                    "irecv", (nonBlockReqCount - 1), nonBlockReqCount
                );
            }
            else if (reqType == CommOp::Barrier)
            {
                MPI_Barrier(epComms[0]);
            }
            else if (reqType == CommOp::AlltoAll ||
                     reqType == CommOp::AlltoAllv)
            {
                if (buf == retBuf)
                {
                    if (reqType == CommOp::AlltoAll)
                        MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf,
                                     length, dataType, epComms[0]);
                    else
                        MPI_Alltoallv(MPI_IN_PLACE, NULL, NULL, MPI_DATATYPE_NULL, buf,
                                      rcvCounts, rcvOffsets, dataType, epComms[0]);
                }
                else
                {
                    for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                    {
                        size_t recvRank = (myRank - procIdx + numprocs) % numprocs;
                        START_COMM(
                            int tag = (myRank + recvRank + reqIdx);
                            size_t epIdx = tag % epCount;
                            MPI_Irecv((char*)retBuf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType, recvRank,
                                       tag, epComms[epIdx], &nonBlockReqs[reqIdx]),
                            "irecv", allToAllSplitParts * procIdx, allToAllSplitParts * (procIdx + 1)
                        );
                    }
                    size_t idxOffset = numprocs * allToAllSplitParts;
                    for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                    {
                        size_t sendRank = (myRank + procIdx + numprocs) % numprocs;
                        START_COMM(
                            int tag = (myRank + sendRank + reqIdx - idxOffset);
                            size_t epIdx = tag % epCount;
                            MPI_Isend((char*)buf + reqOffsets[reqIdx], reqCounts[reqIdx], dataType, sendRank,
                                      tag, epComms[epIdx], &nonBlockReqs[reqIdx]),
                            "isend", idxOffset + allToAllSplitParts * procIdx, idxOffset + allToAllSplitParts * (procIdx + 1)
                        );
                    }
                }
            }
            else
                MLSL_ASSERT(0, "reqType %d is not supported", reqType);

            bp->bufPtr = bp->isInPlace ? buf : retBuf;
            bp->isStarted = true;
            bp->isCompleted = false;

            MLSL_LOG(DEBUG, "rank %zu: op %d: StartComm(%s:%s, buf=%p, len=%zu)",
                     myIdx, bp->GetDesc()->GetOpUniqueId(), bp->GetDesc()->GetCompName().c_str(),
                     bp->GetDesc()->GetOp(0)->GetReqName().c_str(), buf, length);

            return 0;
        }

        void* Wait(bool resetPtr)
        {
            MLSL_LOG(DEBUG, "rank %zu: op %d: Wait(%s:%s, buf=%p, len=%zu, isStarted=%s)",
                     myIdx, bp->GetDesc()->GetOpUniqueId(), bp->GetDesc()->GetCompName().c_str(),
                     (bp->GetDesc()->GetOpCount() > 0 ? bp->GetDesc()->GetOp(0)->GetReqName().c_str() : "Nil"),
                     bp->bufPtr, length, bp->isStarted ? "true" : "false");

            if (bp->isStarted)
                WAIT_COMM(nonBlockReqCount, nonBlockReqs, MPI_STATUSES_IGNORE);

            bp->isStarted = false;
            bp->isCompleted = true;

            void* retPtr = bp->bufPtr;
            if (resetPtr) bp->bufPtr = NULL;
            MLSL_LOG(DEBUG, "Wait: retBuf %p", retPtr);
            return retPtr;
        }

        void* Test(bool* isCompleted, bool resetPtr)
        {
            MLSL_LOG(DEBUG, "rank %zu: op %d: Test(%s:%s, buf=%p, len=%zu, isStarted=%s)",
                     myIdx, bp->GetDesc()->GetOpUniqueId(), bp->GetDesc()->GetCompName().c_str(),
                     (bp->GetDesc()->GetOpCount() > 0 ? bp->GetDesc()->GetOp(0)->GetReqName().c_str() : "Nil"),
                     bp->bufPtr, length, bp->isStarted ? "true" : "false");

            *isCompleted = false;
            if (bp->isCompleted)
            {
                MLSL_LOG(DEBUG, "request is already completed but there is attempt to test it again (completed %d, started %d)",
                         bp->isCompleted, bp->isStarted);

                *isCompleted = true;

                void* retPtr = bp->bufPtr;
                if (resetPtr) bp->bufPtr = NULL;
                return retPtr;
            }

            if (!bp->isStarted)
            {
                MLSL_LOG(DEBUG, "request isn't started but there is attempt to test it, start req fistly (completed %d, started %d)",
                         bp->isCompleted, bp->isStarted);
                return NULL;
            }

            int flag = 1;
            TEST_COMM(nonBlockReqCount, nonBlockReqs, flag, MPI_STATUSES_IGNORE);
            if (!flag) return NULL;

            *isCompleted = true;

            bp->isStarted = false;
            bp->isCompleted = true;

            void* retPtr = bp->bufPtr;
            if (resetPtr) bp->bufPtr = NULL;
            return retPtr;
        }
    };

    void CommRequest::Setup()
    {
        GetImpl()->Setup();
    }

    int CommRequest::Start(void* buf, void* retBuf)
    {
        return GetImpl()->Start(buf, retBuf);
    }

    void* CommRequest::Wait(bool resetPtr)
    {
        return GetImpl()->Wait(resetPtr);
    }

    void* CommRequest::Test(bool *isCompleted, bool resetPtr)
    {
        return GetImpl()->Test(isCompleted, resetPtr);
    }

    void CommRequest::SetCompressionType(CompressionType compressType) {}

    int CommInit(int* argc, char** argv[])
    {
        int provided;
        int isMpiFinalized = 0;
        MPI_Finalized(&isMpiFinalized);
        MLSL_ASSERT(!isMpiFinalized, "MPI_Finalize has been already called, can't restart MPI");

        char* maxShortMsgSizeEnv = NULL;
        if ((maxShortMsgSizeEnv = getenv(MLSL_MAX_SHORT_MSG_SIZE_ENV)) != NULL)
            maxShortMsgSize = atoi(maxShortMsgSizeEnv);

        char* thpThresholdMbEnv = NULL;
        if ((thpThresholdMbEnv = getenv(MLSL_THP_THRESHOLD_MB_ENV)) != NULL)
            thpThresholdMb = atoi(thpThresholdMbEnv);

        char* allToAllSplitEnv = NULL;
        if ((allToAllSplitEnv = getenv(MLSL_ALLTOALL_SPLIT_ENV)) != NULL)
            allToAllSplitParts = atoi(allToAllSplitEnv);
        allToAllSplitParts = MAX(allToAllSplitParts, 1);

        char* threadCountEnv = NULL;
        if ((threadCountEnv = getenv(MLSL_NUM_SERVERS_ENV)) != NULL)
            threadCount = atoi(threadCountEnv);

        MLSL_ASSERT(CHECK_RANGE(threadCount, 0, (THREAD_COUNT_MAX + 1)), "set %s in [0-%zu] range",
                    MLSL_NUM_SERVERS_ENV, (size_t)THREAD_COUNT_MAX);

        char* useMpiForceEnv = NULL;
        if ((useMpiForceEnv = getenv(MLSL_USE_MPI_FORCE_ENV)) != NULL)
            useMpiForce = atoi(useMpiForceEnv);

        char* serverAffinityEnv = NULL;

        char mpiVersion[MPI_MAX_LIBRARY_VERSION_STRING];
        int resultLen;
        MPI_Get_library_version(mpiVersion, &resultLen);
        MLSL_LOG(INFO, "MPI version: %s", mpiVersion);

        if (strncmp(mpiVersion, EXPECTED_MPI_VERSION, strlen(EXPECTED_MPI_VERSION)) == 0)
        {
            if (threadCount)
            {
                char threadCountStr[1024] = { 0 };
                sprintf(threadCountStr, "%zu", threadCount);
                setenv("I_MPI_ASYNC_PROGRESS", "1", 0);
                setenv("I_MPI_ASYNC_PROGRESS_THREADS", threadCountStr, 0);
                setenv("I_MPI_ASYNC_PROGRESS_ID_KEY", COMM_KEY, 0);
                if ((serverAffinityEnv = getenv(MLSL_SERVER_AFFINITY_ENV)) != NULL)
                    setenv("I_MPI_ASYNC_PROGRESS_PIN", serverAffinityEnv, 0);
            }
            else
            {
                setenv("I_MPI_ASYNC_PROGRESS", "0", 0);
                setenv("I_MPI_THREAD_MODE", "direct", 0);
            }
        }
        else
        {
            threadCount = 0;
            MLSL_ASSERT(useMpiForce, "unexpected MPI version: %s, expected: %s, set %s=1 to skip this check",
                        mpiVersion, EXPECTED_MPI_VERSION, MLSL_USE_MPI_FORCE_ENV);
        }

#ifdef INTERNAL_ENV_UPDATE
        //MPI with libfabric requires FI_PROVIDER_PATH env variable which points to the
        //directory with FI providers. Typically providers are located in the "prov" sub-directory
        //of MPI lib directory. We need to find path to MPI library using dladdr() use
        //it to build path to the providers
        //We also update PATH with path to the near "bin" directory
        //MPI_T_PVAR_ALL_HANDLES symbol is chosen randomly
        Dl_info mpi_sym_info{};
        if(dladdr(static_cast<const void*>(MPI_T_PVAR_ALL_HANDLES), &mpi_sym_info) > 0)
        {
            std::string mpi_path{mpi_sym_info.dli_fname};
            mpi_path = mpi_path.substr(0, mpi_path.find_last_of("/"));

            std::string fi_prov_path(mpi_path);
            //Check if "prov" sub-directory exists and add it to the path
            struct stat path_stat;
            if(stat((fi_prov_path + "/prov").c_str(), &path_stat) == 0)
            {
                if(S_ISDIR(path_stat.st_mode))
                {
                    fi_prov_path += "/prov";
                }
            }
            setenv("FI_PROVIDER_PATH", fi_prov_path.c_str(), 1);

            //update PATH if "/lib" is presented in mpi_path
            size_t install_root_pos = mpi_path.rfind("/lib");
            if(install_root_pos != std::string::npos)
            {
                std::string path_to_bin = mpi_path.substr(0, install_root_pos);
                if(stat((path_to_bin + "/bin").c_str(), &path_stat) == 0)
                {
                    if(S_ISDIR(path_stat.st_mode))
                    {
                        path_to_bin += "/bin";
                        std::string common_path = std::getenv("PATH");
                        if(common_path.length() > 0)
                        {
                            common_path += ":";
                        }
                        common_path += path_to_bin;
                        if(stat((path_to_bin + "/process").c_str(), &path_stat) == 0)
                        {
                            if(S_ISDIR(path_stat.st_mode))
                            {
                                common_path += ":";
                                common_path += path_to_bin + "/process";
                            }
                        }
                        setenv("PATH", common_path.c_str(), 1);
                    }
                }
            }
        }
        else
        {
            //Should never happen since libmpi.so must be loaded on start of the binary
            MLSL_ASSERT(0, "Failed to locate libmpi");
        }
#endif

        commCount = MAX(threadCount, 1);

        /* Initialize MPI */
        int ret = MPI_SUCCESS;
        int isMpiInited = 0;
        MPI_Initialized(&isMpiInited);

        if (!isMpiInited)
        {
            ret = MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
            MLSL_ASSERT(provided == MPI_THREAD_MULTIPLE, "unexpected thread level, provided %d", provided);
        }
        else
        {
            MLSL_LOG(INFO, "MPI_Init has been called prior to MLSL::Init");
            isExternalInit = 1;
        }

        globalPg = new ProcessGroupImpl(MPI_COMM_WORLD);
        selfPg = new ProcessGroupImpl(MPI_COMM_SELF);
        myIdx = globalPg->GetBackP()->GetIdx();
        mySize = globalPg->GetBackP()->GetSize();

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            MLSL_LOG(INFO, "%s = %s, actual value %zu",
                     MLSL_NUM_SERVERS_ENV, STR_OR_NULL(threadCountEnv), threadCount);
            MLSL_LOG(INFO, "%s = %s",
                     MLSL_SERVER_AFFINITY_ENV, STR_OR_NULL(serverAffinityEnv));
            MLSL_LOG(INFO, "%s = %s, actual value %zu",
                     MLSL_MAX_SHORT_MSG_SIZE_ENV, STR_OR_NULL(maxShortMsgSizeEnv), maxShortMsgSize);
            MLSL_LOG(INFO, "%s = %s, actual value %zu",
                     MLSL_THP_THRESHOLD_MB_ENV, STR_OR_NULL(thpThresholdMbEnv), thpThresholdMb);
            MLSL_LOG(INFO, "%s = %s, actual value %zu",
                     MLSL_ALLTOALL_SPLIT_ENV, STR_OR_NULL(allToAllSplitEnv), allToAllSplitParts);
            MLSL_LOG(INFO, "%s = %s, actual value %d",
                     MLSL_USE_MPI_FORCE_ENV, STR_OR_NULL(useMpiForceEnv), useMpiForce);
        }

        return ret;
    }

    int CommFinalize()
    {
        if (globalPg)
        {
            delete globalPg;
            globalPg = NULL;
        }

        if (selfPg)
        {
            delete selfPg;
            selfPg = NULL;
        }

        int isMpiFinalized = 0;
        MPI_Finalized(&isMpiFinalized);
        MLSL_ASSERT(!isMpiFinalized, "MPI_Finalize has been already called");

        int isMpiInited = 0;
        MPI_Initialized(&isMpiInited);
        if (isMpiInited)
        {
            MPI_Barrier(MPI_COMM_WORLD);

            if (!isExternalInit)
                return MPI_Finalize();
            else
            {
                MLSL_LOG(INFO, "MPI_Init has been called externally, skip MPI_Finalize");
                return 0;
            }
        }

        return 0;
    }

    void CommBarrier()
    {
        MPI_Barrier(globalPg->GetComm());
    }

    ProcessGroup* GetGlobalProcessGroup()
    {
        return globalPg->GetBackP();
    }

    ProcessGroup* GetSelfProcessGroup()
    {
        return selfPg->GetBackP();
    }

    ProcessGroup* SplitProcessGroup(ProcessGroup* pg, int color)
    {
        MPI_Comm comm;
        MPI_Comm_split(pg->GetImpl()->GetComm(), color, myIdx, &comm);
        ProcessGroupImpl* p = new ProcessGroupImpl(comm);
        return p->GetBackP();
    }

    void SetGlobalProcessGroup(ProcessGroup* pg)
    {
        globalPg = new ProcessGroupImpl(pg->GetImpl()->GetComm());
        myIdx = globalPg->GetBackP()->GetIdx();
        mySize = globalPg->GetBackP()->GetSize();
    }

    ProcessGroup* CreateProcessGroup(int color)
    {
        MPI_Comm comm;
        MPI_Comm_split(globalPg->GetComm(), color, myIdx, &comm);
        ProcessGroupImpl* p = new ProcessGroupImpl(comm);
        return p->GetBackP();
    }

    void FreeProcessGroup(ProcessGroup* pg)
    {
        if (pg == NULL) return;
        ProcessGroupImpl* p = pg->GetImpl();
        delete p;
    }

    void* CommAlloc(size_t sz, size_t alignment)
    {
        void* ptr = NULL;

        if ((sz >= thpThresholdMb * ONE_MB) && (TWO_MB % alignment == 0))
        {
            MLSL_LOG(TRACE, "use THP, size %zu", sz);
            ptr = MLSL_MALLOC(sz, TWO_MB);
        }
        else
        {
            ptr = MLSL_MALLOC(sz, alignment);
        }

        MLSL_LOG(DEBUG, "[%zu] CommAlloc(p=%p, sz=%ld)", myIdx, ptr, sz);
        MLSL_ASSERT(ptr, "NULL pointer");
        return ptr;
    }

    void CommFree(void* ptr)
    {
        MLSL_LOG(DEBUG, "[%zu] CommFree(p=%p)", myIdx, ptr);
        MLSL_FREE(ptr);
    }

    CommRequest* CommCreateRequest(DataType dataType, int opUniqueId, CommDesc::CompType compType)
    {
        CommRequestImpl* p = new CommRequestImpl(dataType, opUniqueId, compType);
        return p->GetBackP();
    }

    void CommFreeRequest(CommRequest* req)
    {
        if (req != NULL) delete req->GetImpl();
    }
};
