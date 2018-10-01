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
/* Optimized comm implementation for MLSL */

#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>

#include "comm.hpp"
#include "common.hpp"
#include "eplib.h"

#if ENABLE_CHKP_INT
#include "pointer_checker.hpp"
#endif

#include "quant.h"

#define EP_NUM_LIMIT            16

#define MAX_EP_PARAM_LEN        4
#define HEAP_SIZE_STR_LEN       5
#define SERVER_AFFINITY_STR_LEN 128
#define EPLIB_ROOT_STR_LEN      4096
#define DYNAMIC_SERVER_STR_LEN  12

#define EPLIB_ROOT_SUFFIX         "/intel64/bin"

/* MLSL->EPLIB env vars mapping */

#define MLSL_DYNAMIC_SERVER_ENV         "MLSL_DYNAMIC_SERVER"
#define MLSL_NUM_SERVERS_ENV            "MLSL_NUM_SERVERS"
#define MLSL_ROOT_ENV                   "MLSL_ROOT"
#define MLSL_HEAP_SIZE_GB_ENV           "MLSL_HEAP_SIZE_GB"
#define MLSL_SERVER_AFFINITY_ENV        "MLSL_SERVER_AFFINITY"
#define MLSL_HOSTNAME_ENV               "MLSL_HOSTNAME"
#define MLSL_HOSTNAME_TYPE_ENV          "MLSL_HOSTNAME_TYPE"
#define MLSL_IFACE_NAME_ENV             "MLSL_IFACE_NAME"
#define MLSL_IFACE_IDX_ENV              "MLSL_IFACE_IDX"
#define MLSL_SERVER_CREATION_TYPE_ENV   "MLSL_SERVER_CREATION_TYPE"
#define MLSL_THP_THRESHOLD_MB_ENV       "MLSL_THP_THRESHOLD_MB"
#define MLSL_CHECK_MEM_SIZE_ENV         "MLSL_CHECK_MEM_SIZE"
#define MLSL_SERVER_PREFIX_ENV          "MLSL_SERVER_PREFIX"
#define MLSL_MSG_PRIORITY_ENV           "MLSL_MSG_PRIORITY"
#define MLSL_MSG_PRIORITY_THRESHOLD_ENV "MLSL_MSG_PRIORITY_THRESHOLD"
#define MLSL_MSG_PRIORITY_MODE_ENV      "MLSL_MSG_PRIORITY_MODE"

#define EPLIB_DYNAMIC_SERVER_ENV         "EPLIB_DYNAMIC_SERVER"
#define EPLIB_MAX_EP_PER_TASK_ENV        "EPLIB_MAX_EP_PER_TASK"
#define EPLIB_ROOT_ENV                   "EPLIB_ROOT"
#define EPLIB_SHM_SIZE_GB_ENV            "EPLIB_SHM_SIZE_GB"
#define EPLIB_SERVER_AFFINITY_ENV        "EPLIB_SERVER_AFFINITY"
#define EPLIB_HOSTNAME_ENV               "EPLIB_HOSTNAME"
#define EPLIB_HOSTNAME_TYPE_ENV          "EPLIB_HOSTNAME_TYPE"
#define EPLIB_IFACE_NAME_ENV             "EPLIB_IFACE_NAME"
#define EPLIB_IFACE_IDX_ENV              "EPLIB_IFACE_IDX"
#define EPLIB_SERVER_CREATION_TYPE_ENV   "EPLIB_SERVER_CREATION_TYPE"
#define EPLIB_THP_THRESHOLD_MB_ENV       "EPLIB_THP_THRESHOLD_MB"
#define EPLIB_CHECK_MEM_SIZE_ENV         "EPLIB_CHECK_MEM_SIZE"
#define EPLIB_SERVER_PREFIX_ENV          "EPLIB_SERVER_PREFIX"
#define EPLIB_MSG_PRIORITY_ENV           "EPLIB_MSG_PRIORITY"
#define EPLIB_MSG_PRIORITY_THRESHOLD_ENV "EPLIB_MSG_PRIORITY_THRESHOLD"
#define EPLIB_MSG_PRIORITY_MODE_ENV      "EPLIB_MSG_PRIORITY_MODE"


/* MLSL env vars */

#define MLSL_MAX_SHORT_MSG_SIZE_ENV "MLSL_MAX_SHORT_MSG_SIZE"
#define MLSL_LARGE_MSG_SIZE_MB_ENV  "MLSL_LARGE_MSG_SIZE_MB"
#define MLSL_LARGE_MSG_CHUNKS_ENV   "MLSL_LARGE_MSG_CHUNKS"
#define MLSL_USE_COPY_THREADS_ENV   "MLSL_USE_COPY_THREADS"
#define OMP_NUM_THREADS_ENV         "OMP_NUM_THREADS"
#define MLSL_COPY_THREADS_ENV       "MLSL_COPY_THREADS"
#define MLSL_COPY_THRESHOLD_ENV     "MLSL_COPY_THRESHOLD"
#define MLSL_CHECK_SINGLE_NODE_ENV  "MLSL_CHECK_SINGLE_NODE"
#define MLSL_ALLTOALL_SPLIT_ENV     "MLSL_ALLTOALL_SPLIT"
#define MLSL_ALLTOALLV_SPLIT_ENV    "MLSL_ALLTOALLV_SPLIT"

#define STR_OR_NULL(str) ((str) ? str : "null")

#define MAX_SHORT_MSG_SIZE    0
#define LARGE_MSG_SIZE_MB     128
#define LARGE_MSG_CHUNK_COUNT 4

#define GET_EP_PAYLOAD(idx, num, count, chunk, istart)                \
  do {                                                                \
      unsigned long iend;                                             \
      unsigned long leftover = count % num;                           \
      chunk = count / num;                                            \
      if (idx < leftover)                                             \
      {                                                               \
          istart = (chunk + 1) * idx;                                 \
          iend = istart + chunk;                                      \
      }                                                               \
      else                                                            \
      {                                                               \
          istart = (chunk + 1) * leftover + chunk * (idx - leftover); \
          iend = istart + chunk - 1;                                  \
      }                                                               \
      chunk = iend - istart + 1;                                      \
  } while (0)

namespace MLSL
{
    ProcessGroupImpl* globalPg = NULL;
    ProcessGroupImpl* selfPg = NULL;
    size_t myIdx;
    size_t mySize;
    size_t epNum = 4;
    size_t epCurrent = 0;
    int epSplit;

    size_t maxShortMsgSize = MAX_SHORT_MSG_SIZE;
    size_t largeMsgSizeMb = LARGE_MSG_SIZE_MB;
    size_t largeMsgChunkCount = LARGE_MSG_CHUNK_COUNT;

    int useCopyThreads = 0;
    size_t copyThreads = 16;
    size_t copyThreshold = 4096;
    int alltoall_split = 0;
    int alltoallv_split = 0;

#if ENABLE_CHKP_INT
    PointerChecker pointerChecker;
#endif

    int is_external_init = 0;


    class ProcessGroupImpl
    {
    private:
        ProcessGroup* bp;
        MPI_Comm comm;
        MPI_Comm* epComm;

        ProcessGroupImpl(const ProcessGroupImpl& processGroup);
        ProcessGroupImpl& operator=(const ProcessGroupImpl& processGroup);

    public:
        ProcessGroupImpl(MPI_Comm comm_) : comm(comm_)
        {
            size_t idx = 0, size = 0;
            MPI_Comm_rank(comm, (int*)&idx);
            MPI_Comm_size(comm, (int*)&size);
            bp = new ProcessGroup(this, idx, size);

            MLSL_LOG(DEBUG, "ProcessGroupImpl: idx %zu, size %zu", idx, size);
            if (size == 1 || epNum == 0)
            {
                int epCommNum =  epNum > 0 ? epNum : 1;
                epComm = (MPI_Comm*)MLSL_MALLOC(epCommNum * sizeof(MPI_Comm), CACHELINE_SIZE);
                MLSL_ASSERT(epComm, "NULL pointer");
                epComm[0] = comm;
                for (size_t epIdx = 1; epIdx < epNum; epIdx++)
                    MPI_Comm_split(comm, 0, epIdx, &epComm[epIdx]);
            }
            else
            {
                epComm = (MPI_Comm*)MLSL_MALLOC(epNum * sizeof(MPI_Comm), CACHELINE_SIZE);
                MLSL_ASSERT(epComm, "NULL pointer");

                MPI_Comm_create_endpoints(comm, epNum, MPI_INFO_NULL, epComm);
            }
        }

        ~ProcessGroupImpl()
        {
            if (bp)
            {
                delete bp;
                bp = NULL;
            }

            if (epComm)
            {
                MLSL_FREE(epComm);
                epComm = NULL;
            }
        }

        MPI_Comm GetComm() { return comm; }
        MPI_Comm* GetEpComm() { return epComm; }
        size_t GetEpNum() { return epNum; }
        ProcessGroup* GetBackP() { return bp; }
    };

    class CommRequestImpl
    {
    private:
        bool isNullReq;
        CommRequest* bp;
        MPI_Datatype dataType;
        size_t dataTypeSize;
        MPI_Op redOp;
        MPI_Comm comm;
        size_t length;
        CommDesc::CompType compType;
        CommOp::ReqType reqType;
        MPI_Comm* epComm;
        size_t epSize;
        size_t epCurr;
        MPI_Request* nonBlockReqs;
        size_t nonBlockReqCount;
        size_t rootIdx;
        size_t myRank;
        size_t numprocs;
        bool useEp;
        size_t groupSize;
        size_t* scounts;
        size_t* rcounts;
        size_t* soffsets;
        size_t* roffsets;
        int* sndcounts;
        int* rcvcounts;
        int* sndoffsets;
        int* rcvoffsets;
        CompressionType compressType;

        struct BufReplacement
        {
            void* sendBuf;
            void* recvBuf;
            void* replaceBuf;
            void* originBuf;
            size_t size;
            int isInPlace;
            int logLevel;

            BufReplacement(void* sbuf, void* rbuf, void* replbuf, void* obuf, size_t s, int inplace, int ll)
                : sendBuf(sbuf), recvBuf(rbuf), replaceBuf(replbuf), originBuf(obuf), size(s), isInPlace(inplace), logLevel(ll)
            {}
        };
        BufReplacement replacement;

        void SetMPIType(DataType dType)
        {
            if (dType == DT_DOUBLE)
            {
                dataType = MPI_DOUBLE;
                dataTypeSize = 8;
            }
            else if (dType == DT_FLOAT)
            {
                dataType = MPI_FLOAT;
                dataTypeSize = 4;
            }
            else if (dType == DT_BYTE)
            {
                dataType = MPI_CHAR;
                dataTypeSize = 1;
            }
            else if (dType == -1)
            {

            }
            else
                MLSL_ASSERT(0, "unsupported datatype %d", dType);
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
                        : isNullReq(false),
                          redOp(MPI_OP_NULL),
                          comm(MPI_COMM_NULL),
                          length(0),
                          compType(CommDesc::NONE),
                          reqType(CommOp::None),
                          epComm(NULL),
                          epSize(0),
                          epCurr(0),
                          nonBlockReqs(NULL),
                          nonBlockReqCount(0),
                          rootIdx(0),
                          myRank(0),
                          numprocs(0),
                          useEp(true),
                          groupSize(0),
                          scounts(NULL),
                          rcounts(NULL),
                          soffsets(NULL),
                          roffsets(NULL),
                          sndcounts(NULL),
                          rcvcounts(NULL),
                          sndoffsets(NULL),
                          rcvoffsets(NULL),
                          compressType(CompressionType::CT_NONE),
                          replacement(NULL, NULL, NULL, NULL, 0, 0, DEBUG)
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

            if (replacement.sendBuf)
            {
                CommFree(replacement.sendBuf);
                replacement.sendBuf = NULL;
            }

            if (replacement.recvBuf)
            {
                CommFree(replacement.recvBuf);
                replacement.recvBuf = NULL;
            }
        }

        void SetCompressionType(CompressionType compressionType)
        {
            compressType = compressionType;
        }

        CommRequest* GetBackP() { return bp; }

        inline void CopyBuf(void* dst, const void* src, size_t bytes)
        {
            MLSL_LOG(DEBUG, "regular memcpy for msg: dst %p, src %p, bytes %zu", dst, src, bytes);
            memcpy(dst, src, bytes);
        }

        void ReplaceIn(void** originSendBuf, void** originRecvBuf)
        {
            if (!useEp) return;

            MLSL_ASSERT(originSendBuf && originRecvBuf, "send and recv buffers should be not null");
            MLSL_LOG(replacement.logLevel, "ReplaceIn: enter");

            replacement.isInPlace = ((*originSendBuf == *originRecvBuf) || reqType == CommOp::Bcast);
            size_t procCount = bp->GetDesc()->GetOp(0)->GetProcessGroup()->GetSize();
            MLSL_LOG(replacement.logLevel, "ReplaceIn: req %s, origin_send_buffer (%p), origin_recv_buffer (%p), is_in_place %d, proc_count %zu",
                                           bp->GetDesc()->GetOp(0)->GetReqName().c_str(), *originSendBuf, *originRecvBuf,
                                           replacement.isInPlace, procCount);

            if (!EPLIB_memory_is_shmem(*originSendBuf))
            {
                MLSL_LOG(replacement.logLevel, "ReplaceIn: origin_send_buffer isn't registered");
                size_t size = length * dataTypeSize;
                if (reqType == CommOp::ReduceScatter ||
                    reqType == CommOp::AlltoAll      ||
                    reqType == CommOp::AllGather     ||
                    reqType == CommOp::Gather        ||
                    reqType == CommOp::Scatter)
                    size *= procCount;

                if (reqType == CommOp::AlltoAllv)
                {
                   size = 0;
                   for(size_t idx=0; idx < numprocs; idx++)
                       size += scounts[idx];
                   size *= dataTypeSize;
                }

                if (reqType == CommOp::AllGatherv && replacement.isInPlace)
                {
                   size = 0;
                   for(size_t idx=0; idx < numprocs; idx++)
                       size += rcounts[idx];
                   size *= dataTypeSize;
                }

                if (!replacement.sendBuf)
                {
                    replacement.sendBuf = CommAlloc(size, CACHELINE_SIZE);
                    MLSL_LOG(replacement.logLevel, "ReplaceIn: alloc send_buffer (%p): calculated_size %zu, tmp_size %zu, length %zu, dtsize %zu",
                                                   replacement.sendBuf, size, bp->tmpSz, length, dataTypeSize);
                }

                if (replacement.isInPlace && reqType != CommOp::AlltoAll && reqType != CommOp::AlltoAllv && reqType != CommOp::AllGatherv)
                {
                    MLSL_ASSERT(((*originSendBuf == *originRecvBuf) || (reqType == CommOp::AllGather) || (reqType == CommOp::Bcast)),
                               "origin_send_buffer (%p) and origin_recv_buffer (%p) should be the same, req_type %s",
                               *originSendBuf, *originRecvBuf, bp->GetDesc()->GetOp(0)->GetReqName().c_str());

                    MLSL_ASSERT(!replacement.originBuf, "origin_buf should be null");
                    MLSL_ASSERT(!replacement.replaceBuf, "replace_buf should be null");
                    MLSL_ASSERT(!replacement.size, "size should be zero");
                    MLSL_LOG(replacement.logLevel, "ReplaceIn: use send_buffer (%p) as replace_buffer", replacement.sendBuf);
                    replacement.originBuf = *originSendBuf;
                    replacement.replaceBuf = replacement.sendBuf;
                    replacement.size = size;
                    if (originRecvBuf) *originRecvBuf = replacement.sendBuf;
                }
                MLSL_LOG(replacement.logLevel, "ReplaceIn: got non registered origin_send_buffer (%p), copy to send_buffer (%p), size %zu",
                                               *originSendBuf, replacement.sendBuf, size);

                CopyBuf(replacement.sendBuf, *originSendBuf, size);
                MLSL_LOG(replacement.logLevel, "ReplaceIn: after memcpy");

                *originSendBuf = replacement.sendBuf;
            }

            if (!EPLIB_memory_is_shmem(*originRecvBuf) && !replacement.isInPlace)
            {
                MLSL_LOG(replacement.logLevel, "ReplaceIn: origin_recv_buffer isn't registered");
                MLSL_LOG(replacement.logLevel, "lenght %zu, dataTypeSize %zu", length, dataTypeSize);
                size_t size = length * dataTypeSize;
                if (reqType == CommOp::ReduceScatter ||
                    reqType == CommOp::AlltoAll      ||
                    reqType == CommOp::AllGather     ||
                    reqType == CommOp::Gather)
                    size *= procCount;

                if (reqType == CommOp::AlltoAllv || reqType == CommOp::AllGatherv)
                {
                  size = 0;
                  for(size_t idx=0; idx < numprocs; idx++)
                      size += rcounts[idx];
                  size *= dataTypeSize;
                }

                if (!replacement.recvBuf)
                {
                    replacement.recvBuf = CommAlloc(size, CACHELINE_SIZE);
                    MLSL_LOG(replacement.logLevel, "ReplaceIn: alloc recv_buffer (%p): calculated_size %zu, tmp_size %zu",
                                                   replacement.recvBuf, size, bp->tmpSz);
                }

                MLSL_ASSERT(!replacement.originBuf, "origin_buf should be null");
                MLSL_ASSERT(!replacement.replaceBuf, "replace_buf should be null");
                MLSL_ASSERT(!replacement.size, "size shouldn't be zero");
                MLSL_LOG(replacement.logLevel, "ReplaceIn: use recv_buffer (%p) as replace_buffer", replacement.recvBuf);
                replacement.originBuf = *originRecvBuf;
                replacement.replaceBuf = replacement.recvBuf;
                replacement.size = size;

                MLSL_LOG(replacement.logLevel, "ReplaceIn: got non registered origin_recv_buffer (%p), copy to recv_buffer (%p), size %zu",
                                               *originRecvBuf, replacement.recvBuf, size);

                CopyBuf(replacement.recvBuf, *originRecvBuf, replacement.size);
                *originRecvBuf = replacement.recvBuf;
            }

            /* create temp recv buf for alltoall over pt2pt */
            if (reqType == CommOp::AlltoAll && replacement.isInPlace)
            {
                size_t size = length * dataTypeSize * procCount;

                if (!replacement.recvBuf)
                {
                    replacement.recvBuf = CommAlloc(size, CACHELINE_SIZE);
                    MLSL_LOG(replacement.logLevel, "ReplaceIn: alloc recv_buffer (%p): calculated_size %zu, tmp_size %zu",
                                                   replacement.recvBuf, size, bp->tmpSz);
                }

                MLSL_LOG(replacement.logLevel, "ReplaceIn: use recv_buffer (%p) as replace_buffer", replacement.recvBuf);
                replacement.originBuf = *originRecvBuf;
                replacement.replaceBuf = replacement.recvBuf;
                replacement.size = size;

                MLSL_LOG(replacement.logLevel, "ReplaceIn: got non registered origin_recv_buffer (%p), copy to recv_buffer (%p), size %zu",
                                               *originRecvBuf, replacement.recvBuf, size);

                CopyBuf(replacement.recvBuf, *originRecvBuf, replacement.size);
                *originRecvBuf = replacement.recvBuf;
            }

            /* create temp recv buf for alltoallv over pt2pt */
            if ((reqType == CommOp::AlltoAllv || reqType == CommOp::AllGatherv) && replacement.isInPlace)
            {
                size_t size = 0;
                for (size_t idx=0; idx < procCount; idx++)
                {
                  size += rcounts[idx];
                }
                size *= dataTypeSize;

                if (!replacement.recvBuf)
                {
                    replacement.recvBuf = CommAlloc(size, CACHELINE_SIZE);
                    MLSL_LOG(replacement.logLevel, "ReplaceIn: alloc recv_buffer (%p): calculated_size %zu, tmp_size %zu",
                                                   replacement.recvBuf, size, bp->tmpSz);
                }

                MLSL_LOG(replacement.logLevel, "ReplaceIn: use recv_buffer (%p) as replace_buffer", replacement.recvBuf);
                replacement.originBuf = *originRecvBuf;
                replacement.replaceBuf = replacement.recvBuf;
                replacement.size = size;

                MLSL_LOG(replacement.logLevel, "ReplaceIn: got non registered origin_recv_buffer (%p), copy to recv_buffer (%p), size %zu",
                                               *originRecvBuf, replacement.recvBuf, size);

                CopyBuf(replacement.recvBuf, *originRecvBuf, replacement.size);
                *originRecvBuf = replacement.recvBuf;
            }
        }

        void ReplaceOut()
        {
            if (!useEp) return;

            MLSL_LOG(replacement.logLevel, "ReplaceOut: enter");

            /* copy from temp recv buf */
            if (replacement.originBuf && replacement.isInPlace && (reqType == CommOp::AlltoAll || reqType == CommOp::AlltoAllv || reqType == CommOp::AllGatherv))
            {
                MLSL_ASSERT(replacement.replaceBuf, "replace_buf should be not null");
                MLSL_LOG(replacement.logLevel, "ReplaceOut: replace origin_buffer (%p) by buffer (%p), size %zu",
                                               replacement.originBuf, replacement.replaceBuf, replacement.size);

                MLSL_ASSERT(replacement.recvBuf == replacement.replaceBuf, "replace_buf and recv_buf should be the same");

                CopyBuf(replacement.originBuf, replacement.replaceBuf, replacement.size);
                bp->bufPtr = replacement.originBuf;
                replacement.originBuf = replacement.replaceBuf = NULL;
                replacement.size = 0;
            }

            if (bp->isStarted && replacement.originBuf && !EPLIB_memory_is_shmem(replacement.originBuf))
            {
                MLSL_ASSERT(replacement.replaceBuf, "replace_buf should be not null");
                MLSL_LOG(replacement.logLevel, "ReplaceOut: replace origin_buffer (%p) by buffer (%p), size %zu",
                                               replacement.originBuf, replacement.replaceBuf, replacement.size);

                if (replacement.isInPlace)
                    MLSL_ASSERT(replacement.sendBuf == replacement.replaceBuf, "replace_buf and send_buf should be the same");
                else
                    MLSL_ASSERT(replacement.recvBuf == replacement.replaceBuf, "replace_buf and recv_buf should be the same");

                CopyBuf(replacement.originBuf, replacement.replaceBuf, replacement.size);
                bp->bufPtr = replacement.originBuf;
                replacement.originBuf = replacement.replaceBuf = NULL;
                replacement.size = 0;
            }
        }

        void Setup()
        {
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
            reqType = op->GetReqType();
            epComm = op->GetProcessGroup()->GetImpl()->GetEpComm();
            epSize = op->GetProcessGroup()->GetImpl()->GetEpNum();
            epCurr = 0;
            groupSize = op->GetProcessGroup()->GetSize();

            MPI_Comm_rank(comm, (int*)&myRank);
            MPI_Comm_size(comm, (int*)&numprocs);

            if (epSize == 0)
                epSplit = 0;
            else
            {
                epSplit = 1;
            }

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
                rcounts = aOp->GetRecvCounts();
                roffsets = new size_t[op->GetProcessGroup()->GetSize()];
                size_t bufSize = rcounts[0];
                roffsets[0] = 0;
                for (int i = 1; i < (int) op->GetProcessGroup()->GetSize(); i++)
                {
                     bufSize += rcounts[i];
                     roffsets[i] = roffsets[i-1] + rcounts[i-1];
                }
                bp->tmpSz = bufSize * dataTypeSize;

                nonBlockReqCount = op->GetProcessGroup()->GetSize();
            }
            else if (reqType == CommOp::ReduceScatter)
            {
                CommOpReduceScatter* aOp = static_cast<CommOpReduceScatter*>(op);
                length = aOp->GetLen();
                SetMPIOp(aOp->GetOp());
                /* In-place not supported when messages split across endpoints */
                if (epSplit == 1) bp->isInPlace = false;
#ifdef USE_OOPRS
                bp->isInPlace = false;
#endif
                if (bp->isInPlace)
                    bp->tmpSz = length * op->GetProcessGroup()->GetSize() * dataTypeSize;
                else
                    bp->tmpSz = length * (op->GetProcessGroup()->GetSize() + 1) * dataTypeSize;
                /* One request per process */
                nonBlockReqCount = op->GetProcessGroup()->GetSize();
            }
            else if (reqType == CommOp::AllReduce)
            {
                CommOpAllReduce* aOp = static_cast<CommOpAllReduce*>(op);
                length = aOp->GetLen();
                SetMPIOp(aOp->GetOp());
                size_t bufSize = length * dataTypeSize;
                bp->tmpSz = bufSize;

                /* Divide equally across all endpoints */
                if ((bufSize >= largeMsgSizeMb * ONE_MB) && (epSize > 0))
                {
                    nonBlockReqCount = epSize * largeMsgChunkCount;
                    MLSL_LOG(DEBUG, "use extra splitting on chunks: ep_size %zu, size %zu, large_msg_size %zu, chunks %zu",
                             epSize, bufSize, largeMsgSizeMb * ONE_MB, nonBlockReqCount);
                }
                else
                    nonBlockReqCount = epSize;
            }
            else if (reqType == CommOp::Reduce)
            {
                CommOpReduce* aOp = static_cast<CommOpReduce*>(op);
                length = aOp->GetLen();
                SetMPIOp(aOp->GetOp());
                rootIdx = aOp->GetRootIdx();
                bp->tmpSz = length * dataTypeSize;
                /* Divide equally across all endpoints */
                nonBlockReqCount = epSize;
            }
            else if (reqType == CommOp::Gather)
            {
                CommOpGather* aOp = static_cast<CommOpGather*>(op);
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                bp->isInPlace = false;
                /* Divide equally across all endpoints */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() + 1;
            }
            else if (reqType == CommOp::Scatter)
            {
                CommOpScatter* aOp = static_cast<CommOpScatter*>(op);
                length = aOp->GetLen();
                rootIdx = aOp->GetRootIdx();
                /* Divide equally across all endpoints */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() + 1;
            }
            else if (reqType == CommOp::Bcast)
            {
                CommOpBcast* aOp = static_cast<CommOpBcast*>(op);
                rootIdx = aOp->GetRootIdx();
                length = aOp->GetLen();
                bp->tmpSz = length * dataTypeSize;
                /* Divide equally across all endpoints */
                nonBlockReqCount = epSize;
            }
            else if (reqType == CommOp::AlltoAll)
            {
                CommOpAlltoAll* aOp = static_cast<CommOpAlltoAll*>(op);
                length = aOp->GetLen();
                /* In-place not supported when messages split across endpoints */
                if (epSplit == 1) bp->isInPlace = false;
#ifdef USE_OOPAA
                bp->isInPlace = false;
#endif
                if (bp->isInPlace)
                    bp->tmpSz = length*op->GetProcessGroup()->GetSize() * dataTypeSize;
                else
                    bp->tmpSz = length*(op->GetProcessGroup()->GetSize() * 2) * dataTypeSize;
                /* Two requests per process - alltoall using send/recv */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() * 2;
                if (alltoall_split)
                    nonBlockReqCount *= epSize;
            }
            else if (reqType == CommOp::AlltoAllv)
            {
                CommOpAlltoAllv* aOp = static_cast<CommOpAlltoAllv*>(op);
                length = 0;

                scounts = aOp->GetSendCounts();
                rcounts = aOp->GetRecvCounts();
                soffsets = aOp->GetSendOffsets();
                roffsets = aOp->GetRecvOffsets();
                /* In-place not supported when messages split across endpoints */
                if (epSplit == 1) bp->isInPlace = false;
#ifdef USE_OOPAA
                bp->isInPlace = false;
#endif
                for (size_t i = 0; i < numprocs; i++)
                  length += rcounts[i];

                bp->tmpSz = length * dataTypeSize;

                /* Two requests per process - alltoallv using send/recv */
                nonBlockReqCount = op->GetProcessGroup()->GetSize() * 2;

                if (alltoallv_split)
                    nonBlockReqCount *= epSize;
            }
            else if (reqType == CommOp::Barrier)
            {
                length = 0;
            }
            else
                MLSL_ASSERT(0, "reqType %d is not supported yet", reqType);

            if (epSplit == 0)
            {
                if (op->GetProcessGroup()->GetImpl()->GetEpNum() != 0)
                {
                    epCurr = epCurrent % op->GetProcessGroup()->GetImpl()->GetEpNum();
                    epCurrent++;
                }
                nonBlockReqCount = 1;
            }

            nonBlockReqs = (MPI_Request*)MLSL_MALLOC(nonBlockReqCount * sizeof(MPI_Request), CACHELINE_SIZE);
            MLSL_LOG(DEBUG, "nonBlockReqs %p, nonBlockReqCount %zu", nonBlockReqs, nonBlockReqCount);
            MLSL_ASSERT(nonBlockReqs, "nonBlockReqs is NULL, nonBlockReqCount %zu", nonBlockReqCount);
            for (size_t idx = 0; idx < nonBlockReqCount; idx++) nonBlockReqs[idx] = MPI_REQUEST_NULL;

            useEp = (length > maxShortMsgSize) ? true : false;
            if (reqType == CommOp::Bcast     ||
                reqType == CommOp::AllReduce ||
                reqType == CommOp::Reduce)
                useEp = useEp && (length >= epNum);
            useEp = useEp && (epSplit == 1);

        }

        int Start(void* buf, void* retBuf)
        {
            if (isNullReq) return 0;

            ReplaceIn(&buf, &retBuf);
            if (reqType == CommOp::AllGather)
            {
                if (buf == retBuf)
                {
                    if (useEp)
                    {
                        nonBlockReqCount = numprocs;
                        for (size_t idx = 0; idx < nonBlockReqCount; idx++)
                        {
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + length * dataTypeSize * idx, length * dataTypeSize);
#endif
                            MPI_Ibcast((char*)buf + length * dataTypeSize * idx, length, dataType,
                                       idx, epComm[idx % epSize], &nonBlockReqs[idx]);
                        }
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
#endif
                        MPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf,
                                       length, dataType, comm, &nonBlockReqs[0]);
                    }
                }
                else
                {
#if ENABLE_CHKP_INT
                    pointerChecker.Check(buf, length * dataTypeSize);
                    pointerChecker.Check(retBuf, length * groupSize * dataTypeSize);
#endif
                    MPI_Iallgather(buf, length, dataType,
                                   retBuf, length, dataType,
                                   comm, &nonBlockReqs[0]);
                }
            }
            else if (reqType == CommOp::AllGatherv)
            {
                if (buf == retBuf)
                {
                     if (useEp)
                     {
                         nonBlockReqCount = numprocs;
                         for (size_t idx = 0; idx < nonBlockReqCount; idx++)
                         {
#if ENABLE_CHKP_INT
                             pointerChecker.Check((char*)buf + roffsets[idx] * dataTypeSize, rcount[idx] * dataTypeSize);
#endif
                             MPI_Ibcast((char*)buf + roffsets[idx] * dataTypeSize, rcounts[idx], dataType,
                                        idx, epComm[idx % epSize], &nonBlockReqs[idx]);
                         }
                     }
                     else
                     {
                         int* tmp_recvCounts = new int[numprocs];
                         int* tmp_recvOffsets = new int[numprocs];
                         size_t total_count = 0;
                         for (int i = 0; i < (int) numprocs; i++)
                         {
                             tmp_recvCounts[i]  = (int) rcounts[i];
                             tmp_recvOffsets[i] = (int) roffsets[i];

                             total_count += tmp_recvCounts[i];
                         }
#if ENABLE_CHKP_INT
                         pointerChecker.Check(buf, total_count * groupSize * dataTypeSize);
#endif
                         MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf,
                                        tmp_recvCounts, tmp_recvOffsets, dataType, comm, &nonBlockReqs[0]);
                     }
                }
                else
                {
                     if (useEp)
                     {
                         memcpy((char*) retBuf + roffsets[myRank] * dataTypeSize, buf, rcounts[myRank] * dataTypeSize);
                         nonBlockReqCount = numprocs;
                         for (size_t idx = 0; idx < nonBlockReqCount; idx++)
                         {
#if ENABLE_CHKP_INT
                             pointerChecker.Check((char*)retBuf + roffsets[idx] * dataTypeSize, rcounts[idx] * dataTypeSize);
#endif
                             MPI_Ibcast((char*)retBuf + roffsets[idx] * dataTypeSize, rcounts[idx], dataType,
                                        idx, epComm[idx % epSize], &nonBlockReqs[idx]);
                         }
                     }
                     else
                     {
#if ENABLE_CHKP_INT
                         pointerChecker.Check(buf, length * dataTypeSize);
                         pointerChecker.Check(retBuf, length * groupSize * dataTypeSize);
#endif
                         int* tmp_recvCounts = new int[numprocs];
                         int* tmp_recvOffsets = new int[numprocs];
                         for (int i = 0; i < (int) numprocs; i++)
                         {
                             tmp_recvCounts[i]  = (int) rcounts[i];
                             tmp_recvOffsets[i] = (int) roffsets[i];
                         }
                         MPI_Iallgatherv(buf, length, dataType,
                                    retBuf, tmp_recvCounts, tmp_recvOffsets, dataType,
                                    comm, &nonBlockReqs[0]);
                     }
                }
            }
            else if (reqType == CommOp::Bcast)
            {
                if (useEp)
                {
                    long start, chunk;
                    for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                    {
                        GET_EP_PAYLOAD(reqIdx, nonBlockReqCount, length, chunk, start);

#if ENABLE_CHKP_INT
                        pointerChecker.Check((char*)buf + start * dataTypeSize, chunk * dataTypeSize);
#endif
                        MPI_Ibcast((char*)buf + start * dataTypeSize, chunk, dataType,
                                  rootIdx, epComm[epCurr + reqIdx], &nonBlockReqs[reqIdx]);
                    }
                }
                else
                {

#if ENABLE_CHKP_INT
                    pointerChecker.Check(buf, length * dataTypeSize);
#endif
                    MPI_Ibcast((char*)buf, length, dataType,
                              rootIdx, comm, &nonBlockReqs[0]);
                }
            }
            else if (reqType == CommOp::ReduceScatter)
            {
                if (useEp)
                {
                    for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check((char*)buf + length * dataTypeSize * reqIdx, length * dataTypeSize);
#endif
                        MPI_Ireduce((char*)buf + length * dataTypeSize * reqIdx, retBuf, length, dataType,
                                    redOp, reqIdx, epComm[epCurr + reqIdx % epSize], &nonBlockReqs[reqIdx]);
                    }
                }
                else
                {
                    if (bp->isInPlace)
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
#endif
                        MPI_Ireduce_scatter_block(MPI_IN_PLACE, buf, length, dataType,
                                                  redOp, comm, &nonBlockReqs[0]);
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
                        pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                        MPI_Ireduce_scatter_block(buf, retBuf, length, dataType,
                                                  redOp, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::AllReduce)
            {
                if (buf == retBuf)
                {
                    if (useEp)
                    {
                        long start, chunk;
                        MPI_Op realRedOp = redOp;
                        if (compressType == CompressionType::CT_QUANTIZATION)
                        {
                            MLSL_ASSERT(numprocs > 1, "quantization is not supported with process count = 1");
                            realRedOp = MPI_QUANT_OP;
                        }

                        for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                        {
                            GET_EP_PAYLOAD(reqIdx, nonBlockReqCount, length, chunk, start);
                            MLSL_LOG(TRACE, "allreduce (in-place): len %zu, chunk size %ld, chunk count %zu", length, chunk, nonBlockReqCount);
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + start * dataTypeSize, chunk * dataTypeSize);
#endif
                            MPI_Iallreduce(MPI_IN_PLACE, (char*)buf + start * dataTypeSize, chunk, dataType,
                                           realRedOp, epComm[(epCurr + reqIdx) % epNum], &nonBlockReqs[reqIdx]);
                        }
                    }
                    else
                    {
                        if (compressType != CompressionType::CT_NONE)
                            MLSL_LOG(DEBUG, "quantization is not supported qith low length or whit MLSL_NUM_SERVERS < 1");
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * dataTypeSize);
#endif
                        MPI_Iallreduce(MPI_IN_PLACE, buf, length, dataType,
                                       redOp, comm, &nonBlockReqs[0]);
                    }
                }
                else
                {
                    if (useEp)
                    {
                        long start, chunk;
                        MPI_Op realRedOp = redOp;
                        if (compressType == CompressionType::CT_QUANTIZATION)
                        {
                            MLSL_ASSERT(numprocs > 1, "quantization is not supported with process count = 1");
                            realRedOp = MPI_QUANT_OP;
                        }

                        for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                        {
                            GET_EP_PAYLOAD(reqIdx, nonBlockReqCount, length, chunk, start);
                            MLSL_LOG(TRACE, "allreduce (out-of-place): len %zu, chunk size %ld, chunk count %zu", length, chunk, nonBlockReqCount);
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + start * dataTypeSize, chunk * dataTypeSize);
                            pointerChecker.Check((char*)retBuf + start * dataTypeSize, chunk * dataTypeSize);
#endif
                            MPI_Iallreduce((char*)buf + start * dataTypeSize, (char*)retBuf + start * dataTypeSize, chunk, dataType,
                                           realRedOp, epComm[(epCurr + reqIdx) % epNum], &nonBlockReqs[reqIdx]);
                        }
                    }
                    else
                    {
                        if (compressType != CompressionType::CT_NONE)
                            MLSL_LOG(DEBUG, "quantization is not supported qith low length or whit MLSL_NUM_SERVERS < 1");
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * dataTypeSize);
                        pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                        MPI_Iallreduce(buf, retBuf, length, dataType,
                                       redOp, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::Reduce)
            {
                if (buf == retBuf)
                {
                    if (useEp)
                    {
                        long start, chunk;
                        for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                        {
                            GET_EP_PAYLOAD(reqIdx, nonBlockReqCount, length, chunk, start);
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + start * dataTypeSize, chunk * dataTypeSize);
#endif
                            if (myRank == rootIdx)
                                MPI_Ireduce(MPI_IN_PLACE, (char*)buf + start * dataTypeSize, chunk, dataType,
                                            redOp, rootIdx, epComm[epCurr + reqIdx], &nonBlockReqs[reqIdx]);
                            else
                                MPI_Ireduce((char*)buf + start * dataTypeSize, (char*)buf + start * dataTypeSize, chunk, dataType,
                                            redOp, rootIdx, epComm[epCurr + reqIdx], &nonBlockReqs[reqIdx]);
                        }
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * dataTypeSize);
#endif
                        if (myRank == rootIdx)
                            MPI_Ireduce(MPI_IN_PLACE, buf, length, dataType,
                                        redOp, rootIdx, comm, &nonBlockReqs[0]);
                        else
                            MPI_Ireduce(buf, buf, length, dataType,
                                        redOp, rootIdx, comm, &nonBlockReqs[0]);
                    }
                }
                else
                {
                    if (useEp)
                    {
                        long start, chunk;
                        for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                        {
                            GET_EP_PAYLOAD(reqIdx, nonBlockReqCount, length, chunk, start);
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + start * dataTypeSize, chunk * dataTypeSize);
                            pointerChecker.Check((char*)retBuf + start * dataTypeSize, chunk * dataTypeSize);
#endif
                            MPI_Ireduce((char*)buf + start * dataTypeSize, (char*)retBuf + start * dataTypeSize, chunk, dataType,
                                        redOp, rootIdx, epComm[epCurr + reqIdx], &nonBlockReqs[reqIdx]);

                        }
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * dataTypeSize);
                        pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                        MPI_Ireduce(buf, retBuf, length, dataType, redOp, rootIdx, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::Gather)
            {
                if (useEp)
                {
                    if (myRank == rootIdx)
                    {
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (rootIdx - procIdx + numprocs - 1) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)retBuf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            MPI_Irecv((char*)retBuf + length * dataTypeSize * j, length, dataType,
                                      j, 20000, epComm[(rootIdx + j) % epSize], &nonBlockReqs[procIdx]);
                        }
                    }
                    size_t j = myRank;
#if ENABLE_CHKP_INT
                    pointerChecker.Check(buf, length * dataTypeSize);
#endif
                    MPI_Isend((char*)buf, length, dataType,
                              rootIdx, 20000, epComm[(rootIdx + j) % epSize], &nonBlockReqs[numprocs]);
                }
                else
                {
                    if (buf == retBuf)
                    {
                        if (myRank == rootIdx)
                        {
#if ENABLE_CHKP_INT
                            pointerChecker.Check(buf, length * groupSize * dataTypeSize);
#endif
                            MPI_Igather(MPI_IN_PLACE, length, dataType, buf,
                                        length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                        }
                        else
                        {
#if ENABLE_CHKP_INT
                            pointerChecker.Check(buf, length * dataTypeSize);
                            pointerChecker.Check(retBuf, length * groupSize * dataTypeSize);
#endif
                            MPI_Igather(buf, length, dataType, retBuf,
                                        length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                        }
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * dataTypeSize);
                        pointerChecker.Check(retBuf, length * groupSize * dataTypeSize);
#endif
                        MPI_Igather(buf, length, dataType, retBuf,
                                    length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::Scatter)
            {
                if (useEp)
                {
                    if (myRank == rootIdx)
                    {
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (rootIdx - procIdx + numprocs - 1) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            MPI_Isend((char*)buf + length * dataTypeSize * j, length, dataType,
                                    j, 19999, epComm[(rootIdx + j) % epSize], &nonBlockReqs[procIdx]);
                        }
                    }
                    size_t j = myRank;
#if ENABLE_CHKP_INT
                    pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                    MPI_Irecv((char*)retBuf, length, dataType,
                              rootIdx, 19999, epComm[(rootIdx + j) % epSize], &nonBlockReqs[numprocs]);
                }
                else
                {
                    if (buf == retBuf)
                    {
                        if (myRank == rootIdx)
                        {
#if ENABLE_CHKP_INT
                            pointerChecker.Check(buf, length * groupSize * dataTypeSize);
#endif
                            MPI_Iscatter(buf, length, dataType, MPI_IN_PLACE,
                                         length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                        }
                        else
                        {
#if ENABLE_CHKP_INT
                            pointerChecker.Check(buf, length * groupSize * dataTypeSize);
                            pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                            MPI_Iscatter(buf, length, dataType, retBuf,
                                         length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                        }
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
                        pointerChecker.Check(retBuf, length * dataTypeSize);
#endif
                        MPI_Iscatter(buf, length, dataType, retBuf,
                                     length, dataType, rootIdx, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::Barrier)
            {
                MPI_Barrier(comm);
            }
            else if (reqType == CommOp::AlltoAll)
            {
                if (useEp)
                {
                    if (alltoall_split)
                    {
                        long start, chunk;
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank - procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)retBuf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            for (size_t epIdx=0; epIdx < epSize; epIdx++)
                            {
                                GET_EP_PAYLOAD(epIdx, epSize, length, chunk, start);
                                MPI_Irecv((char*)retBuf + (start + j * length) * dataTypeSize, chunk, dataType,
                                        j, (myRank + j), epComm[epIdx], &nonBlockReqs[procIdx + numprocs * epIdx]);
                            }
                        }
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank + procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            for (size_t epIdx=0; epIdx < epSize; epIdx++)
                            {
                                GET_EP_PAYLOAD(epIdx, epSize, length, chunk, start);
                                MPI_Isend((char*)buf + (start + j * length) * dataTypeSize, chunk, dataType,
                                        j, (myRank + j), epComm[epIdx], &nonBlockReqs[numprocs * (epSize + epIdx) + procIdx]);
                            }
                        }
                    }
                    else
                    {
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank - procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)retBuf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            MPI_Irecv((char*)retBuf + length * dataTypeSize * j, length, dataType, 
                                    j, (myRank + j), epComm[(myRank + j) % epSize], &nonBlockReqs[procIdx + numprocs]);
                        }
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank + procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + length * dataTypeSize * j, length * dataTypeSize);
#endif
                            MPI_Isend((char*)buf + length * dataTypeSize * j, length, dataType,
                                    j, (myRank + j), epComm[(myRank + j) % epSize], &nonBlockReqs[procIdx]);
                        }
                    }

                }
                else
                {
                    if (buf == retBuf)
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
#endif
                        MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf,
                                      length, dataType, comm, &nonBlockReqs[0]);
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, length * groupSize * dataTypeSize);
                        pointerChecker.Check(retBuf, length * groupSize * dataTypeSize);
#endif
                        MPI_Ialltoall(buf, length, dataType, retBuf,
                                      length, dataType, comm, &nonBlockReqs[0]);
                    }
                }
            }
            else if (reqType == CommOp::AlltoAllv)
            {
                if (useEp)
                {
                    if (alltoallv_split)
                    {
                        long start, chunk;
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank - procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)retBuf + roffsets[j] * dataTypeSize, rcounts[j] * dataTypeSize);
#endif
                            for (size_t epIdx=0; epIdx < epSize; epIdx++)
                            {
                                GET_EP_PAYLOAD(epIdx, epSize, rcounts[j], chunk, start);
                                MPI_Irecv((char*)retBuf + (roffsets[j] + start) * dataTypeSize, chunk, dataType,
                                        j, (myRank + j), epComm[epIdx], &nonBlockReqs[procIdx + numprocs * epIdx]);
                            }
                        }
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank + procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + soffsets[j] * dataTypeSize, scounts[j] * dataTypeSize);
#endif
                            for (size_t epIdx=0; epIdx < epSize; epIdx++)
                            {
                                GET_EP_PAYLOAD(epIdx, epSize, scounts[j], chunk, start);
                                MPI_Isend((char*)buf + (soffsets[j] + start) * dataTypeSize, chunk, dataType,
                                        j, (myRank + j), epComm[epIdx], &nonBlockReqs[numprocs * (epSize + epIdx) + procIdx]);
                            }
                        }
                    }
                    else
                    {
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank - procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)retBuf + roffsets[j] * dataTypeSize, rcounts[j] * dataTypeSize);
#endif
                            MPI_Irecv((char*)retBuf + roffsets[j] * dataTypeSize, rcounts[j], dataType,
                                    j, (myRank + j), epComm[(myRank + j) % epSize], &nonBlockReqs[procIdx + numprocs]);
                        }
                        for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                        {
                            size_t j = (myRank + procIdx + numprocs) % numprocs;
#if ENABLE_CHKP_INT
                            pointerChecker.Check((char*)buf + soffsets[j] * dataTypeSize, scounts[j] * dataTypeSize);
#endif
                            MPI_Isend((char*)buf + soffsets[j] * dataTypeSize, scounts[j], dataType,
                                    j, (myRank + j), epComm[(myRank + j) % epSize], &nonBlockReqs[procIdx]);
                        }
                    }
                }
                else
                {
#if ENABLE_CHKP_INT
                    size_t total_send_len = 0;
                    size_t total_recv_len = 0;
                    for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                    {
                        total_send_len += scount[procIdx];
                        total_recv_len += rcount[procIdx];
                    }
#endif
                    rcvcounts = (int*)malloc(sizeof(int)*numprocs);
                    rcvoffsets = (int*)malloc(sizeof(int)*numprocs);
                    sndcounts = (int*)malloc(sizeof(int)*numprocs);
                    sndoffsets = (int*)malloc(sizeof(int)*numprocs);
                    MLSL_ASSERT(rcvcounts && rcvoffsets && sndcounts && sndoffsets, "Can't allocate memory");
                    for (size_t procIdx = 0; procIdx < numprocs; procIdx++)
                    {
                        rcvcounts[procIdx] = rcounts[procIdx];
                        rcvoffsets[procIdx] = roffsets[procIdx];
                        sndcounts[procIdx] = scounts[procIdx];
                        sndoffsets[procIdx] = soffsets[procIdx];
                    }

                    if (buf == retBuf)
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, total_recv_len * dataTypeSize);
#endif
                        MPI_Ialltoallv(MPI_IN_PLACE, NULL, NULL, MPI_DATATYPE_NULL, buf,
                                      rcvcounts, rcvoffsets, dataType, comm, &nonBlockReqs[0]);
                    }
                    else
                    {
#if ENABLE_CHKP_INT
                        pointerChecker.Check(buf, total_send_len * dataTypeSize);
                        pointerChecker.Check(retBuf, total_recv_len * dataTypeSize);
#endif
                        MPI_Ialltoallv(buf, sndcounts, sndoffsets, dataType, retBuf,
                                      rcvcounts, rcvoffsets, dataType, comm, &nonBlockReqs[0]);
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
            {
                for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                {
                    MLSL_LOG(DEBUG, "[%zu] size %zu nonBlockReqCount %zu reqIdx %zu request: (%ld, %p)",
                             myIdx, mySize, nonBlockReqCount, reqIdx, (long) nonBlockReqs[reqIdx], &nonBlockReqs[reqIdx]);
                    MPI_Wait(&nonBlockReqs[reqIdx], MPI_STATUS_IGNORE);
                    nonBlockReqs[reqIdx] = MPI_REQUEST_NULL;
                }
            }

            ReplaceOut();

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
            for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
            {
                MPI_Test(&nonBlockReqs[reqIdx], &flag, MPI_STATUS_IGNORE);
                if (!flag) return NULL;
            }

            *isCompleted = true;

            for (size_t reqIdx = 0; reqIdx < nonBlockReqCount; reqIdx++)
                nonBlockReqs[reqIdx] = MPI_REQUEST_NULL;

            ReplaceOut();

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

    void CommRequest::SetCompressionType(CompressionType compressType)
    {
        GetImpl()->SetCompressionType(compressType);
    }

    void SetWorkarounds()
    {
        /* FIXME: there are extra communicators which should be explicitly deleted
                  to remove shm_coll* files from /dev/shm; find these communicators and delete over MPI_Comm_free.
                  Temporarily disable shm-aware collectives to avoid that problem. */
        setenv("I_MPI_COLL_INTRANODE", "pt2pt", 0);

        /* Enabling of DAPL memory registration cache leads to correctness issues on collective functional tests 
           where we allocate/deallocate buffers very frequently */
        setenv("I_MPI_DAPL_TRANSLATION_CACHE", "0", 0);
    }

    int CommInit(int* argc, char** argv[])
    {
        int is_mpi_finalized = 0;
        MPI_Finalized(&is_mpi_finalized);
        MLSL_ASSERT(!is_mpi_finalized, "MPI_Finalize has been already called, can't restart MPI");

        SetWorkarounds();

        /* Initialize MPI */
        int ret = MPI_SUCCESS;
        int is_mpi_inited = 0;
        MPI_Initialized(&is_mpi_inited);

        char* dynamic_server_env = NULL;
        if (!is_mpi_inited)
        {
            dynamic_server_env = getenv(MLSL_DYNAMIC_SERVER_ENV);
            if (dynamic_server_env != NULL)
            {
                setenv(EPLIB_DYNAMIC_SERVER_ENV, dynamic_server_env, 0);
                if (strncmp(dynamic_server_env, "asyncthread", DYNAMIC_SERVER_STR_LEN) == 0)
                {
                    int provided;
                    ret = PMPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
                    MLSL_ASSERT((provided != MPI_THREAD_MULTIPLE),
                               "Requested thread level not provided.\n");
                }
                else
                    ret = PMPI_Init(argc, argv);
            }
            else
            {
                /* FIXME: the work around for MLSL_NUM_SERVERS=1 case (dynamic server thread), remove after fix on EPLIB side */
                setenv(EPLIB_DYNAMIC_SERVER_ENV, "process", 0);
                ret = PMPI_Init(argc, argv);
            }
        }
        else
        {
            /* We don't plan to support this scenario
             * Preferably only MLSL::Init needs to be called
             */
            MLSL_LOG(INFO, "MPI_Init has been called prior to MLSL::Init");
            setenv(EPLIB_DYNAMIC_SERVER_ENV, "process", 0);
            is_external_init = 1;
        }

        /* Process env vars */
        /* Map MLSL env vars to EPLIB env vars */

        char* max_short_msg_size_env = NULL;
        if ((max_short_msg_size_env = getenv(MLSL_MAX_SHORT_MSG_SIZE_ENV)) != NULL)
            maxShortMsgSize = atoi(max_short_msg_size_env);

        char* large_msg_size_mb_env = NULL;
        if ((large_msg_size_mb_env = getenv(MLSL_LARGE_MSG_SIZE_MB_ENV)) != NULL)
            largeMsgSizeMb = atoi(large_msg_size_mb_env);

        char* large_msg_chunks_env = NULL;
        if ((large_msg_chunks_env = getenv(MLSL_LARGE_MSG_CHUNKS_ENV)) != NULL)
        {
            largeMsgChunkCount = atoi(large_msg_chunks_env);
            if (!largeMsgChunkCount) largeMsgChunkCount = 1;
        }

        char* use_copy_threads_env = NULL;
        if ((use_copy_threads_env = getenv(MLSL_USE_COPY_THREADS_ENV)) != NULL)
            useCopyThreads = atoi(use_copy_threads_env);

        char* copy_threads_env = NULL;
        if ((copy_threads_env = getenv(OMP_NUM_THREADS_ENV)) != NULL)
            copyThreads = atoi(copy_threads_env);

        if ((copy_threads_env = getenv(MLSL_COPY_THREADS_ENV)) != NULL)
            copyThreads = atoi(copy_threads_env);

        char* copy_threshold_env = NULL;
        if ((copy_threshold_env = getenv(MLSL_COPY_THRESHOLD_ENV)) != NULL)
            copyThreshold = atoi(copy_threshold_env);

        /* EPLIB_MAX_EP_PER_TASK */
        char* num_servers_env = NULL;
        if ((num_servers_env = getenv(MLSL_NUM_SERVERS_ENV)) != NULL)
            epNum = atoi(num_servers_env);

        int rank, comm_size;
        PMPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* Always disable EPLIB for single-process case
           cause we can't launch spawn processes wo process manager */
        if (comm_size == 1) epNum = 0;

        int check_single_node = 1;
        char* check_single_node_env = NULL;
        if ((check_single_node_env = getenv(MLSL_CHECK_SINGLE_NODE_ENV)) != NULL)
            check_single_node = atoi(check_single_node_env);

        if (check_single_node)
        {
            MPI_Comm node_comm;
            PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
            int node_comm_size;
            PMPI_Comm_size(node_comm, &node_comm_size);
            if (comm_size == node_comm_size && !num_servers_env) epNum = 0;
            PMPI_Comm_free(&node_comm);
        }

        MLSL_ASSERT(CHECK_RANGE(epNum, 0, (EP_NUM_LIMIT + 1)), "set %s in [0-%zu] range",
                    MLSL_NUM_SERVERS_ENV, (size_t)EP_NUM_LIMIT);

        char ep_num_env[MAX_EP_PARAM_LEN] = {0};
        snprintf(ep_num_env, MAX_EP_PARAM_LEN, "%zu", epNum);
        setenv(EPLIB_MAX_EP_PER_TASK_ENV, ep_num_env, 0);

        /* EPLIB_ROOT */
        char* mlsl_root_env = NULL;
        char eplib_root_env[EPLIB_ROOT_STR_LEN];
        if ((mlsl_root_env = getenv(MLSL_ROOT_ENV)) != NULL)
            snprintf(eplib_root_env, EPLIB_ROOT_STR_LEN, "%s%s",
                     mlsl_root_env, EPLIB_ROOT_SUFFIX);
        else
            MLSL_ASSERT(0, "specify %s", MLSL_ROOT_ENV);
        setenv(EPLIB_ROOT_ENV, eplib_root_env, 0);

        /* EPLIB_SHM_SIZE_GB */
        char* heap_size_env = NULL;
        if ((heap_size_env = getenv(MLSL_HEAP_SIZE_GB_ENV)) != NULL)
            setenv(EPLIB_SHM_SIZE_GB_ENV, heap_size_env, 0);

        /* EPLIB_SERVER_AFFINITY */
        char* server_affinity_env = NULL;
        if ((server_affinity_env = getenv(MLSL_SERVER_AFFINITY_ENV)) != NULL)
            setenv(EPLIB_SERVER_AFFINITY_ENV, server_affinity_env, 0);

        /* EPLIB_HOSTNAME */
        char* hostname_env = NULL;
        if ((hostname_env = getenv(MLSL_HOSTNAME_ENV)) != NULL)
            setenv(EPLIB_HOSTNAME_ENV, hostname_env, 0);

        /* EPLIB_HOSTNAME_TYPE */
        char* hostname_type_env = NULL;
        if ((hostname_type_env = getenv(MLSL_HOSTNAME_TYPE_ENV)) != NULL)
            setenv(EPLIB_HOSTNAME_TYPE_ENV, hostname_type_env, 0);

        /* EPLIB_IFACE_NAME */
        char* iface_name_env = NULL;
        if ((iface_name_env = getenv(MLSL_IFACE_NAME_ENV)) != NULL)
            setenv(EPLIB_IFACE_NAME_ENV, iface_name_env, 0);

        /* EPLIB_IFACE_IDX */
        char* iface_idx_env = NULL;
        if ((iface_idx_env = getenv(MLSL_IFACE_IDX_ENV)) != NULL)
            setenv(EPLIB_IFACE_IDX_ENV, iface_idx_env, 0);

        /* EPLIB_SERVER_CREATION_TYPE */
        char* server_creation_type_env = NULL;
        if ((server_creation_type_env = getenv(MLSL_SERVER_CREATION_TYPE_ENV)) != NULL)
            setenv(EPLIB_SERVER_CREATION_TYPE_ENV, server_creation_type_env, 0);

        /* EPLIB_THP_THRESHOLD_MB */
        char* thp_threshold_mb_env = NULL;
        if ((thp_threshold_mb_env = getenv(MLSL_THP_THRESHOLD_MB_ENV)) != NULL)
            setenv(EPLIB_THP_THRESHOLD_MB_ENV, thp_threshold_mb_env, 0);

        /* EPLIB_CHECK_MEM_SIZE */
        char* check_mem_size_env = NULL;
        if ((check_mem_size_env = getenv(MLSL_CHECK_MEM_SIZE_ENV)) != NULL)
            setenv(EPLIB_CHECK_MEM_SIZE_ENV, check_mem_size_env, 0);

        /* EPLIB_SERVER_PREFIX */
        char* server_prefix_env = NULL;
        if ((server_prefix_env = getenv(MLSL_SERVER_PREFIX_ENV)) != NULL)
            setenv(EPLIB_SERVER_PREFIX_ENV, server_prefix_env, 0);

        /* MLSL_ALLTOALL_SPLIT */
        char* alltoall_split_env = NULL;
        if ((alltoall_split_env = getenv(MLSL_ALLTOALL_SPLIT_ENV)) != NULL)
            alltoall_split = atoi(alltoall_split_env);

        /* MLSL_ALLTOALLV_SPLIT */
        char* alltoallv_split_env = NULL;
        if ((alltoallv_split_env = getenv(MLSL_ALLTOALLV_SPLIT_ENV)) != NULL)
            alltoallv_split = atoi(alltoallv_split_env);

        /* EPLIB_MSG_PRIORITY */
        char* msg_priority_env = NULL;
        if ((msg_priority_env = getenv(MLSL_MSG_PRIORITY_ENV)) != NULL)
        {
            if ((comm_size & (comm_size - 1)) != 0)
                setenv(EPLIB_MSG_PRIORITY_ENV, "0", 0);
            else
                setenv(EPLIB_MSG_PRIORITY_ENV, msg_priority_env, 0);
        }

        /* EPLIB_MSG_PRIORITY_THRESHOLD */
        char* msg_priority_threshold_env = NULL;
        if ((msg_priority_threshold_env = getenv(MLSL_MSG_PRIORITY_THRESHOLD_ENV)) != NULL)
            setenv(EPLIB_MSG_PRIORITY_THRESHOLD_ENV, msg_priority_threshold_env, 0);

        /* EPLIB_MSG_PRIORITY_MODE */
        char* msg_priority_mode_env = NULL;
        if ((msg_priority_mode_env = getenv(MLSL_MSG_PRIORITY_MODE_ENV)) != NULL)
            setenv(EPLIB_MSG_PRIORITY_MODE_ENV, msg_priority_mode_env, 0);

        if (rank == 0)
        {
            if ((comm_size & (comm_size - 1)) != 0 && msg_priority_env != NULL)
            {
                MLSL_LOG(INFO, "Experimental version of Allreduce algorithm with priority supports only the amount " \
                               "of MPI processes that is equal to a power of two.");
                MLSL_LOG(INFO, "Process count is %d, message prioritization is disabled.", comm_size);
            }

            /* MLSL->EPLIB mapping */
            MLSL_LOG(INFO, "%s = %s", MLSL_DYNAMIC_SERVER_ENV, STR_OR_NULL(dynamic_server_env));
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_NUM_SERVERS_ENV, STR_OR_NULL(num_servers_env), epNum);
            MLSL_LOG(INFO, "%s = %s", MLSL_ROOT_ENV, STR_OR_NULL(mlsl_root_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_HEAP_SIZE_GB_ENV, STR_OR_NULL(heap_size_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_SERVER_AFFINITY_ENV, STR_OR_NULL(server_affinity_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_HOSTNAME_ENV, STR_OR_NULL(hostname_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_HOSTNAME_TYPE_ENV, STR_OR_NULL(hostname_type_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_IFACE_NAME_ENV, STR_OR_NULL(iface_name_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_IFACE_IDX_ENV, STR_OR_NULL(iface_idx_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_SERVER_CREATION_TYPE_ENV, STR_OR_NULL(server_creation_type_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_THP_THRESHOLD_MB_ENV, STR_OR_NULL(thp_threshold_mb_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_CHECK_MEM_SIZE_ENV, STR_OR_NULL(check_mem_size_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_SERVER_PREFIX_ENV, STR_OR_NULL(server_prefix_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_MSG_PRIORITY_ENV, STR_OR_NULL(msg_priority_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_MSG_PRIORITY_THRESHOLD_ENV, STR_OR_NULL(msg_priority_threshold_env));
            MLSL_LOG(INFO, "%s = %s", MLSL_MSG_PRIORITY_MODE_ENV, STR_OR_NULL(msg_priority_mode_env));
            
            /* MLSL env vars */
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_MAX_SHORT_MSG_SIZE_ENV, STR_OR_NULL(max_short_msg_size_env), maxShortMsgSize);
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_LARGE_MSG_SIZE_MB_ENV, STR_OR_NULL(large_msg_size_mb_env), largeMsgSizeMb);
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_LARGE_MSG_CHUNKS_ENV, STR_OR_NULL(large_msg_chunks_env), largeMsgChunkCount);
            MLSL_LOG(INFO, "%s = %s, actual value %d", MLSL_USE_COPY_THREADS_ENV, STR_OR_NULL(use_copy_threads_env), useCopyThreads);
            MLSL_LOG(INFO, "%s = %s", OMP_NUM_THREADS_ENV, getenv(OMP_NUM_THREADS_ENV));
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_COPY_THREADS_ENV, STR_OR_NULL(copy_threads_env), copyThreads);
            MLSL_LOG(INFO, "%s = %s, actual value %zu", MLSL_COPY_THRESHOLD_ENV, STR_OR_NULL(copy_threshold_env), copyThreshold);
            MLSL_LOG(INFO, "%s = %s, actual value %d", MLSL_CHECK_SINGLE_NODE_ENV, STR_OR_NULL(check_single_node_env), check_single_node);
            MLSL_LOG(INFO, "%s = %s, actual value %d", MLSL_ALLTOALL_SPLIT_ENV, STR_OR_NULL(alltoall_split_env), alltoall_split);
            MLSL_LOG(INFO, "%s = %s, actual value %d", MLSL_ALLTOALLV_SPLIT_ENV, STR_OR_NULL(alltoallv_split_env), alltoallv_split);
        }

        /* Initialize EPLIB */
        EPLIB_init();

        globalPg = new ProcessGroupImpl(MPI_COMM_WORLD);
        selfPg = new ProcessGroupImpl(MPI_COMM_SELF);
        myIdx = globalPg->GetBackP()->GetIdx();
        mySize = globalPg->GetBackP()->GetSize();

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


        int is_mpi_finalized = 0;
        MPI_Finalized(&is_mpi_finalized);
        MLSL_ASSERT(!is_mpi_finalized, "MPI_Finalize has been already called");

        int is_mpi_inited = 0;
        MPI_Initialized(&is_mpi_inited);
        if (is_mpi_inited)
        {
            PMPI_Barrier(MPI_COMM_WORLD);
            EPLIB_finalize();
            PMPI_Barrier(MPI_COMM_WORLD);

            if (!is_external_init)
                return PMPI_Finalize();
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
        MPI_Comm comm = p->GetComm();
        MPI_Comm_free(&comm);
        delete p;
    }

    void* CommAlloc(size_t sz, size_t alignment)
    {
        void* ptr = EPLIB_memalign(alignment, sz);
        MLSL_ASSERT(ptr, "NULL pointer");
        MLSL_LOG(DEBUG, "[%zu] CommAlloc(p=%p, sz=%ld)", myIdx, ptr, sz);

#if ENABLE_CHKP_INT
        pointerChecker.Add(ptr, sz);
#endif

        return ptr;
    }

    void CommFree(void* ptr)
    {
        MLSL_LOG(DEBUG, "[%zu] CommFree(p=%p)", myIdx, ptr);

#if ENABLE_CHKP_INT
        pointerChecker.Remove(ptr);
#endif

        EPLIB_free(ptr);
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
