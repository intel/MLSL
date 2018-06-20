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
#ifndef MLSL_IMPL_HPP
#define MLSL_IMPL_HPP

/* Internal DL Specific Interface for MLSL */

#include <set>
#include <string>
#include <vector>

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>

#include "comm.hpp"
#include "common.hpp"
#include "env.hpp"
#include "mlsl.hpp"
#include "eplib.h"

#define STATS_OUTPUT_FILE "mlsl_stats.log"
#define INVALID_OP_IDX    "invalid operation idx"

namespace MLSL
{
    extern int opUniqueId;
    extern pid_t initPid;
    extern ProcessGroup* globalProcessGroup;
    extern ProcessGroup* selfProcessGroup;
    extern QuantParams* globalQuantParam;

    class StatisticsImpl;
    class OperationImpl;
    class DistributionImpl;
    class ActivationImpl;
    class ParameterSetImpl;
    class CommBlockInfoImpl;
    class OperationRegInfoImpl;
    class SessionImpl;

    class RequestStorage
    {
    private:
        RequestStorage() { reqIdx = 0; }
        ~RequestStorage() {}
        RequestStorage(const RequestStorage&);
        RequestStorage& operator=(const RequestStorage&);
        std::set<CommRequest*> requests;
        size_t reqIdx;
        size_t GetNextCommRequestIdx() { return ++reqIdx; }

    public:
        static RequestStorage& GetObject()
        {
            static RequestStorage obj;
            return obj;
        }

        size_t RegisterRequest(CommRequest* req)
        {
            size_t idx = GetNextCommRequestIdx();
            requests.insert(req);
            return idx;
        }

        void RemoveRequest(CommRequest* req)
        {
            set<CommRequest*>::iterator itr = requests.find(req);
            MLSL_ASSERT((itr != requests.end()), "can't find req %p in storage", req);
            requests.erase(itr);
            CommFreeRequest(req);
        }

        size_t GetSize() { return requests.size(); }
    };

    inline int mysnprintf(char* buf, size_t size, const char*  format, ...)
    {
        va_list args;
        va_start (args, format);
        size_t c = vsnprintf(buf, size, format, args);

        if (c > 0 && c < size)
            return c;
        else if (c >= size)
            return (size - 1);
        else
            return 0;
    }

    inline int mysnprintf(FILE* outFile, char *buf, size_t size, const char * format, ...)
    {
        va_list args;
        va_start (args, format);

        size_t c = -1;
        if(outFile != NULL)
            c = vfprintf(outFile, format, args);
        else
            c = vsnprintf(buf, size, format, args);

        if (c > 0 && c < size)
            return c;
        else if (c >= size)
            return size-1;
        else
            return 0;
    }

    class CommBuf
    {
    private:
        void* p;
        size_t size;
        bool isOwned;
    public:
        CommBuf(size_t sz) : p(0), size(sz), isOwned(false) { }

        ~CommBuf() { Free(); }

        size_t GetSize() { return size; }

        size_t Alloc()
        {
            if (size == 0) return 0;
            p = Environment::GetEnv().Alloc(size, CACHELINE_SIZE);
            isOwned = true;
            if (p != NULL) return size;
            return 0;
        }

        void Free()
        {
            if (isOwned && p != 0)
            {
                Environment::GetEnv().Free(p);
                p = 0;
            }
        }

        int SetPtr(void* ptr)
        {
            if (p != 0 && isOwned)
            {
                Environment::GetEnv().Free(p);
                isOwned = false;
            }
            p = ptr;
            return 0;
        }

        void* GetPtr() { return p; }
    };

    class DistributionImpl : public Distribution
    {
    private:
        size_t dataParts;
        size_t modelParts;
        bool replicate;
        size_t replicaCount;
        ProcessGroup* dataProcessGroup;
        ProcessGroup* modelProcessGroup;
        ProcessGroup* replicaProcessGroup;

        DistributionImpl(const DistributionImpl& dist);
        DistributionImpl& operator=(const DistributionImpl& dist);

        ProcessGroup* GetGroupByType(GroupType gt)
        {
            switch (gt)
            {
                case GT_DATA: return dataProcessGroup;
                case GT_MODEL: return modelProcessGroup;
                case GT_GLOBAL: return globalProcessGroup;
                default: MLSL_ASSERT(0, "unexpected group type %d", gt);
            }
        }

        /*Should be removed (ReduceOp and ReductionType is the same) - was used only for fast prototyping*/
        CommOp::ReduceOp GetReduceOpByType(ReductionType rt)
        {
            switch (rt)
            {
                case RT_MIN: return CommOp::RO_MIN;
                case RT_MAX: return CommOp::RO_MAX;
                case RT_SUM: return CommOp::RO_SUM;
                default: MLSL_ASSERT(0, "unexpected reduction type %d", rt);
            }
        }

    public:
        DistributionImpl(size_t dataPartitions, size_t modelPartitions, bool replicate_, int dataColor, int modelColor)
        {
            if (dataColor == -1 && modelColor == -1)
            {
                MLSL_LOG(DEBUG, "dataPartitions %zu, modelPartitions %zu", dataPartitions, modelPartitions);
                MLSL_ASSERT(dataPartitions > 0 && modelPartitions > 0 && (int)dataPartitions > 0 && (int)modelPartitions > 0,
                            "numbers for data and model groups must be positive");

                replicate = replicate_;
                dataParts = dataPartitions;
                modelParts = modelPartitions;

                size_t globalGroupSize = globalProcessGroup->GetSize();
                size_t globalGroupIdx = globalProcessGroup->GetIdx();

                size_t lSize = dataParts * modelParts;
                size_t lId = globalGroupIdx % lSize;
                if (replicate == true)
                    replicaCount = globalGroupSize / lSize;
                else
                    replicaCount = 1;

                size_t replicaColor;
                size_t iR = globalGroupIdx / lSize;
                size_t iM = lId / modelParts;
                size_t iF = lId % modelParts;
                modelColor = int(iR * lSize + iM);
                dataColor = int(iR * lSize + iF);
                replicaColor  = lId;
                if (modelParts == 1)
                    modelProcessGroup = selfProcessGroup;
                else if (modelParts == globalGroupSize)
                    modelProcessGroup = globalProcessGroup;
                else
                    modelProcessGroup = CreateProcessGroup(modelColor);

                if (dataParts == 1)
                    dataProcessGroup = selfProcessGroup;
                else if (dataParts == globalGroupSize)
                {
                    if (envData.dupGroup)
                        dataProcessGroup = CreateProcessGroup(1 /* the same color on all processes */);
                    else
                        dataProcessGroup = globalProcessGroup;
                }
                else
                    dataProcessGroup = CreateProcessGroup(dataColor);

                if (replicaCount == 1)
                    replicaProcessGroup = selfProcessGroup;
                else if (replicaCount == globalGroupSize)
                    replicaProcessGroup = globalProcessGroup;
                else
                    replicaProcessGroup = CreateProcessGroup((int)replicaColor);
            }
            else
            {
                replicate = false;
                replicaCount = 1;
                modelProcessGroup = CreateProcessGroup(modelColor);
                dataProcessGroup = CreateProcessGroup(dataColor);
                replicaProcessGroup = selfProcessGroup;
                
                dataParts = dataProcessGroup->GetSize();
                modelParts = modelProcessGroup->GetSize();
            }
        }

        ~DistributionImpl(void)
        {
            if (modelProcessGroup != globalProcessGroup && modelProcessGroup != selfProcessGroup) FreeProcessGroup(modelProcessGroup);
            if (dataProcessGroup != globalProcessGroup && dataProcessGroup != selfProcessGroup) FreeProcessGroup(dataProcessGroup);
            if (replicaProcessGroup != globalProcessGroup && replicaProcessGroup != selfProcessGroup) FreeProcessGroup(replicaProcessGroup);
            modelProcessGroup = dataProcessGroup = replicaProcessGroup = NULL;
        }

        ProcessGroup* GetDataProcessGroup() { return dataProcessGroup; }
        ProcessGroup* GetModelProcessGroup() { return modelProcessGroup; }

        size_t GetDataParts() { return dataParts; }
        size_t GetModelParts() { return modelParts; }

        void Barrier(GroupType gt);
        CommReq* Bcast(void* buffer, size_t count, DataType dataType, size_t groupRootId, GroupType gt);
        CommReq* Reduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, size_t groupRootId, GroupType gt);
        CommReq* AllReduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, GroupType gt);
        CommReq* AlltoAll(void* sendBuffer, size_t sendCount, void* recvbuffer, DataType dataType, GroupType gt);
        CommReq* AlltoAllv(void* sendBuffer, size_t* sendCounts, size_t* sendOffsets, void* recvBuffer, size_t* recvCounts, size_t* recvOffsets, DataType dataType, GroupType gt);
        CommReq* Gather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, size_t groupRootId, GroupType gt);
        CommReq* AllGather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType groupType);
        CommReq* AllGatherv(void* sendBuffer, size_t sendCount, void* recvBuffer, size_t* recvCounts, DataType dataType, GroupType groupType);
        CommReq* Scatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, size_t groupRootId, GroupType gt);
        CommReq* ReduceScatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, ReductionType rt, GroupType gt);
    };

    class RegActivation
    {
    private:
        size_t count;
        size_t size;
        DataType dataType;

        RegActivation(const RegActivation& regAct);
        RegActivation& operator=(const RegActivation& regAct);

    public:
        RegActivation(size_t actCount, size_t actSize, DataType dType)
                      : count(actCount), size(actSize), dataType(dType) { }
        size_t GetCount() { return count;}
        size_t GetSize() { return size; }
        DataType GetDataType() { return dataType; }
    };

    class RegParameterSet
    {
    private:
        size_t count;
        size_t size;
        DataType dataType;
        bool distributedUpdate;
        CompressionType compressType;

        RegParameterSet(const RegParameterSet& regParamSet);
        RegParameterSet& operator=(const RegParameterSet& regParamSet);

    public:
        RegParameterSet(size_t paramSetCount, size_t paramSetSize, DataType dType, bool distUpdate, CompressionType compressType = CompressionType::CT_NONE)
                       : count(paramSetCount), size(paramSetSize), dataType(dType), distributedUpdate(distUpdate), compressType(compressType) { }
        size_t GetCount() { return count;}
        size_t GetSize() { return size; }
        DataType GetDataType() { return dataType; }
        bool GetDistributedUpdate() { return distributedUpdate; }
        CompressionType GetCompressionType() { return compressType; }
    };

    class OperationRegInfoImpl : public OperationRegInfo
    {
    private:
        OpType opType;
        string name;
        vector<RegActivation*>   inActs;
        vector<RegActivation*>   outActs;
        vector<RegParameterSet*> paramSets;
        size_t refCount;

        OperationRegInfoImpl(const OperationRegInfoImpl& opRegInfo);
        OperationRegInfoImpl& operator=(const OperationRegInfoImpl& opRegInfo);

    public:
        OperationRegInfoImpl(OpType operationType, string name_ = "") : opType(operationType), name(name_)
        {
            refCount = 1;
        }
        void IncrementRefCount()
        {
            refCount++;
        }

        bool CanDelete()
        {
            refCount--;
            if (refCount == 0)
                return true;
            return false;
        }

        ~OperationRegInfoImpl()
        {
            for (size_t actIdx = 0; actIdx < inActs.size(); actIdx++)
                delete inActs[actIdx];

            for (size_t actIdx = 0; actIdx < outActs.size(); actIdx++)
                delete outActs[actIdx];

            for (size_t paramIdx = 0; paramIdx < paramSets.size(); paramIdx++)
                delete paramSets[paramIdx];

            inActs.clear();
            outActs.clear();
            paramSets.clear();
        }

        const char* GetName() { return name.c_str(); }
        OpType GetOpType() { return opType; }
        size_t GetInputCount() { return inActs.size(); }
        size_t GetOutputCount() { return outActs.size(); }
        size_t GetParameterSetCount() { return paramSets.size(); }
        RegActivation* GetInput(size_t idx) { return inActs.at(idx); }
        RegActivation* GetOutput(size_t idx) { return outActs.at(idx); }
        RegParameterSet* GetParameterSet(size_t idx) { return paramSets.at(idx); }

        void SetName(const char* n) { name = string(n); }

        size_t AddInput(RegActivation* act)
        {
            inActs.push_back(act);
            return inActs.size() - 1;
        }

        size_t AddOutput(RegActivation* act)
        {
            outActs.push_back(act);
            return outActs.size() - 1;
        }

        size_t AddParameterSet(RegParameterSet* paramSet)
        {
            paramSets.push_back(paramSet);
            return paramSets.size() - 1;
        }

        void Validate(Distribution* dist __attribute__ ((unused)))
        {
            if (opType == OT_CC) {}
            else if (opType == OT_BIAS) {}
            else if (opType == OT_ACT) {}
            else if (opType == OT_POOL) {}
            else if (opType == OT_DATA) {}
            else if (opType == OT_EVAL) {}
            else if (opType == OT_BCAST) {}
            else if (opType == OT_CONCAT) {}
            else MLSL_ASSERT(0, "opType %d is not supported yet", opType);
        }
    };

    class CommBlockInfoImpl : public CommBlockInfo
    {
    private:
        size_t mbOffset;
        size_t mbCount;
        size_t fmOffset;
        size_t fmCount;
        size_t fmSize;
        DataType dataType;
        size_t bufOffset;

        CommBlockInfoImpl(const CommBlockInfoImpl& bi);
        CommBlockInfoImpl& operator=(const CommBlockInfoImpl& bi);

    public:
        CommBlockInfoImpl(size_t mbOff, size_t mbC, size_t fmOff, size_t fmC, size_t fmSz, DataType dType, size_t bufOff)
            : mbOffset(mbOff), mbCount(mbC), fmOffset(fmOff), fmCount(fmC),
            fmSize(fmSz), dataType(dType), bufOffset(bufOff) {}

        ~CommBlockInfoImpl() {}

        size_t GetMbOffset() { return mbOffset; }
        size_t GetMbCount() { return mbCount; }
        size_t GetFmOffset() { return fmOffset; }
        size_t GetFmCount() { return fmCount; }
        size_t GetFmSize() { return fmSize; }
        DataType GetDataType() { return dataType; }
        size_t GetBufOffset() { return bufOffset; }
    };

    class ActivationImpl : public Activation
    {
    private:
        bool isInput;
        size_t globalFmCount;
        size_t globalFmOffset;
        size_t localFmCount;
        CommBuf* commBuf;
        size_t featureMapSize;
        DataType dataType;
        DistributionImpl* dist;
        OperationImpl* op;
        vector<CommBlockInfoImpl*> packBlocks;
        vector<CommBlockInfoImpl*> unpackBlocks;
        bool needReduce;
        bool needComm;
        CommRequest* commReq;
        size_t tmpBufOffset;
        ActivationImpl* peerAct;
        bool isPeerSet;
        size_t actIndex; /* activation index inside operation */

        void BIPackReduceScatter();
        void BIPackReduceScatter2();
        void BIUnpackReduceScatter();
        void BIPackAllReduce();
        void BIUnpackAllReduce();
        void BIPackAllGather();
        void BIUnpackAllGather();
        void BIUnpackAllGather2();
        void BIBuildAlltoAll(ActivationImpl* inAct);

        ActivationImpl(const ActivationImpl& act);
        ActivationImpl& operator=(const ActivationImpl& act);

    public:
        ActivationImpl(OperationImpl* op, RegActivation* rp, bool isInput_, size_t id_);
        ~ActivationImpl()
        {
            if (commBuf)
            {
                delete commBuf;
                commBuf = NULL;
            }

            if (commReq)
            {
                CommFreeRequest(commReq);
                commReq = NULL;
            }

            for (size_t blockIdx = 0; blockIdx < packBlocks.size(); blockIdx++)
                delete packBlocks.at(blockIdx);

            for (size_t blockIdx = 0; blockIdx < unpackBlocks.size(); blockIdx++)
                delete unpackBlocks.at(blockIdx);

            packBlocks.clear();
            unpackBlocks.clear();
        }

        size_t GetGlobalFmCount() { return globalFmCount; }
        size_t GetGlobalFmOffset() { return globalFmOffset; }
        size_t GetLocalFmCount() { return localFmCount; }
        int SetPeer(ActivationImpl* peerAct);
        int InitPeerConnection();
        ActivationImpl* GetPeer() { return peerAct; }
        bool IsPeerSet() { return isPeerSet; }
        bool IsCommRequired() { return needComm; }
        OperationImpl* GetOp() { return op; }
        size_t GetActIndex() { return actIndex; }
        CommBuf* GetCommBuf() { return commBuf; }
        size_t GetMsgSize()
        {
            if (commReq && commReq->GetDesc() && commReq->GetDesc()->GetOpCount() > 0)
            {
                size_t dataTypeSize = (commReq->GetDesc()->GetDataType() == DT_FLOAT) ? 4 : 8;
                return commReq->GetDesc()->GetOp(0)->GetLen() * dataTypeSize;
            }
            else
                return 0;
        }
        size_t GetPackBlockCount() { return packBlocks.size(); }
        size_t GetUnpackBlockCount() { return unpackBlocks.size(); }
        CommBlockInfo* GetPackBlock(size_t blockIdx) { return packBlocks.at(blockIdx); }
        CommBlockInfo* GetUnpackBlock(size_t blockIdx) { return unpackBlocks.at(blockIdx); }
        DataType GetDataType() { return dataType; }
        size_t GetFmSize() { return featureMapSize; }
        void StartComm(void* buf);
        void* WaitComm();
        int PrintInfo(char* b, size_t m)
        {
            size_t c = 0;
            c += mysnprintf(b + c, m - c, "global_count:%zu local_count:%zu global_offset:%zu need_comm:%s ",
                            globalFmCount, localFmCount, globalFmOffset, needComm ? "true" : "false");
            if (needComm)
            {
                c += mysnprintf(b + c, m - c, "comm_op:%s pack_block_count:%zu unpack_block_count:%zu tmp_size:%zu\n",
                                (commReq->GetDesc()->GetOpCount() > 0 ? commReq->GetDesc()->GetOp(0)->GetReqName().c_str() : "nil"),
                                GetPackBlockCount(), GetUnpackBlockCount(), (commBuf ? commBuf->GetSize() : 0));
                for (size_t blockIdx = 0; blockIdx < GetPackBlockCount(); blockIdx++)
                {
                    CommBlockInfo* bi = GetPackBlock(blockIdx);
                    c += mysnprintf(b + c, m - c, "   pack_block   %zu: mb_offset:%zu mb_count:%zu fm_offset:%zu fm_count:%zu fm_size:%zu buf_offset:%zu\n",
                                    blockIdx, bi->GetMbOffset(), bi->GetMbCount(), bi->GetFmOffset(),
                                    bi->GetFmCount(),bi->GetFmSize(), bi->GetBufOffset());
                }
                for (size_t blockIdx = 0; blockIdx < GetUnpackBlockCount(); blockIdx++)
                {
                    CommBlockInfo* bi = GetUnpackBlock(blockIdx);
                    c += mysnprintf(b + c, m - c, "   unpack_block %zu: mb_offset:%zu mb_count:%zu fm_offset:%zu fm_count:%zu fm_size:%zu buf_offset:%zu\n",
                                    blockIdx, bi->GetMbOffset(), bi->GetMbCount(), bi->GetFmOffset(),
                                    bi->GetFmCount(),bi->GetFmSize(), bi->GetBufOffset());
                }
            }
            else
                c += mysnprintf(b + c, m - c, "\n");
            return c;
        }
    };

    class ParameterSetImpl : public ParameterSet
    {
    protected:
        size_t globalKernelCount;
        size_t globalKernelOffset;
        size_t localKernelCount;
        size_t ownedKernelCount;
        size_t ownedKernelOffset;
        CommBuf* commBuf;
        size_t kernelSize;
        DataType dataType;
        bool needComm;
        bool distributedUpdate;
        CommRequest* gradReq;
        CommRequest* incReq;
        OperationImpl* op;
        DistributionImpl* dist;
        size_t paramIndex; /* param index inside operation */
        CompressionType compType;

        ParameterSetImpl(const ParameterSetImpl& w);
        ParameterSetImpl& operator=(const ParameterSetImpl& w);

    public:
        ParameterSetImpl(OperationImpl* op_, RegParameterSet* rp, size_t id_);
        ~ParameterSetImpl()
        {
            if (commBuf)
            {
                delete commBuf;
                commBuf = NULL;
            }

            if (gradReq)
            {
                CommFreeRequest(gradReq);
                gradReq = NULL;
            }

            if (incReq)
            {
                CommFreeRequest(incReq);
                incReq = NULL;
            }
        }

        size_t GetGlobalKernelCount() { return globalKernelCount; }
        size_t GetGlobalKernelOffset() { return globalKernelOffset; }
        size_t GetLocalKernelCount() { return localKernelCount; }
        size_t GetOwnedKernelCount() { return ownedKernelCount; }
        size_t GetOwnedKernelOffset() { return ownedKernelOffset; }
        bool IsCommRequired() { return needComm; }
        bool IsDistributedUpdate() { return distributedUpdate; }
        CommBuf* GetCommBuf() { return commBuf; }
        size_t GetGradientMsgSize()
        {
            CommRequest* req = gradReq;
            if (req && req->GetDesc() && req->GetDesc()->GetOpCount() > 0)
            {
                size_t dataTypeSize = (req->GetDesc()->GetDataType() == DT_FLOAT) ? 4 : 8;
                return req->GetDesc()->GetOp(0)->GetLen() * dataTypeSize;
            }
            else
                return 0;
        }
        size_t GetIncrementMsgSize()
        {
            CommRequest* req = incReq;
            if (req && req->GetDesc() && req->GetDesc()->GetOpCount() > 0)
            {
                size_t dataTypeSize = (req->GetDesc()->GetDataType() == DT_FLOAT) ? 4 : 8;
                return req->GetDesc()->GetOp(0)->GetLen() * dataTypeSize;
            }
            else
                return 0;
        }
        DataType GetDataType() { return dataType; }
        size_t GetKernelSize() {return kernelSize; }
        void StartGradientComm(void* buf);
        void StartIncrementComm(void* buf);
        void* WaitGradientComm();
        void* TestGradientComm(bool* isCompleted);
        void* WaitIncrementComm();
        size_t PrintInfo(char* b, size_t m)
        {
            size_t c = 0;
            c += mysnprintf(b + c, m - c, "global_count:%zu local_count:%zu global_offset:%zu owned_count:%zu owned_offset:%zu need_comm:%s ",
                            globalKernelCount, localKernelCount, globalKernelOffset, ownedKernelCount,
                            ownedKernelOffset, needComm ? "true" : "false");
            if (needComm)
            {
                c += mysnprintf(b + c, m - c, "[grad: comm_op:%s tmp_size:%zu] ",
                        (gradReq->GetDesc()->GetOpCount() > 0 ? gradReq->GetDesc()->GetOp(0)->GetReqName().c_str() : "nil"),
                        (commBuf ? commBuf->GetSize() : 0));

                if (distributedUpdate)
                    c += mysnprintf(b + c, m - c, "[inc: comm_op:%s]",
                            (incReq->GetDesc()->GetOpCount() > 0 ? incReq->GetDesc()->GetOp(0)->GetReqName().c_str() : "nil"));
                else
                    c += mysnprintf(b + c, m - c, "[inc: comm_op:nil]");
            }
            c += mysnprintf(b + c, m - c, "\n");
            return c;
        }
    };

    struct StatData
    {
        vector<unsigned long long> isolationCommStartTimes;
        vector<unsigned long long> isolationCommWaitTimes;
        vector<unsigned long long> compStartTimes;
        vector<unsigned long long> compWaitTimes;
        vector<unsigned long long> commStartTimes;
        vector<unsigned long long> commWaitTimes;

        vector<size_t> isolationCommSizesPerBatch; /* size per each entity, for one isolation iteration/one minibatch */
        vector<size_t> isolationCommSizes;         /* size per each entity, sum over isolation iterations */
        size_t isolationCommSize;                  /* sum over each entity of operation, sum over isolation iterations */
        unsigned long long isolationCommTime;      /* sum over each entity of operation */

        vector<size_t> commSizes;    /* size per each entity */
        size_t commSize;             /* sum over each entity of operation */
        unsigned long long commTime; /* sum over each entity of operation */
        unsigned long long compTime;

        vector<bool> isTested; /* whether invoked Test at least once for certain entity (ParameterSet or Actiovation) in operation */
    };

    class StatEvent
    {
    public:

        enum ActionType
        {
            Start,
            Wait,
            Test
        };

        size_t opIdx;
        size_t entIdx;
        bool isCompTime;
        bool isParam;
        bool isInputOrIncrement;
        ActionType actionType;

        StatEvent(size_t opIdx, size_t entIdx, bool isCompTime, bool isParam, bool isInputOrIncrement, ActionType actionType) :
                opIdx(opIdx),
                entIdx(entIdx),
                isCompTime(isCompTime),
                isParam(isParam),
                isInputOrIncrement(isInputOrIncrement),
                actionType(actionType)
                {}

        void Print()
        {
            MLSL_LOG(DEBUG, "stat_event: [op_idx: %zu, ent_idx: %zu, comp: %d, param: %d, input_or_incr: %d, action: %d]",
                     opIdx, entIdx, isCompTime, isParam, isInputOrIncrement, actionType);
        }
    };

    class StatisticsImpl : public Statistics
    {
    private:
        SessionImpl* session;
        vector<StatData> stats;
        unsigned long long totalCompTime;
        unsigned long long totalCommTime;
        size_t totalCommSize;

        unsigned long long totalIsolationCommTime;
        size_t totalIsolationCommSize;

        unsigned long long globalTime;
        size_t iterations, skip;
        bool isStarted;
        size_t batchCount;
        int startOpMonitor;
        size_t maxIACount, maxOACount, maxParamCount;
        bool isIACommSet, isOACommSet, isParamCommSet;
        bool isStatsEnabled;

        StatisticsImpl(const StatisticsImpl& st);
        StatisticsImpl& operator=(const StatisticsImpl& st);

        inline double GetGHzFreq()
        {
            unsigned long long startTime = rdtsc();
            sleep(1);
            unsigned long long endTime = rdtsc();
            return (double)(endTime - startTime) / 1.0e9;
        }
        void PrintIsolationComm(FILE* outFile);
        void Print(FILE* outFile);

    public:
        StatisticsImpl(Session* s);
        ~StatisticsImpl() {}

        void CollectIsolationStats();
        void Reset();
        void Initialize();
        void UpdateStats(const StatEvent& event);
        
        void SetGlobalTime(unsigned long long time) { globalTime = time; }
        unsigned long long GetGlobalTime() { return globalTime; }
        int GetStartOpMonitor() { return startOpMonitor; }
        vector<StatData>& GetStatData() { return stats; }
        bool IsStarted() { return isStarted; }
        bool IsEnabled() { return isStatsEnabled; }
        void Stop()
        {
            MLSL_ASSERT(isStarted == true, "start collection before stopping");
            isStarted = false;
        }

        void Start()
        {
            MLSL_ASSERT(isStarted == false, "stop collection before starting");
            isStarted = true;
        }

        void Print()
        {
            MLSL_ASSERT(isStarted == false, "stop collection before printing");

            if (!isStatsEnabled)
                return;

            fclose(fopen(STATS_OUTPUT_FILE, "w"));
            FILE* outputFile = fopen(STATS_OUTPUT_FILE, "a");
            MLSL_ASSERT(outputFile, "outputFile is null");
            Print(outputFile);
            fclose(outputFile);
        }

        unsigned long long GetIsolationCommCycles(size_t opIdx) { MLSL_ASSERT(opIdx < stats.size(), INVALID_OP_IDX); return stats[opIdx].isolationCommTime / (iterations - skip); }
        unsigned long long GetCommCycles(size_t opIdx) { MLSL_ASSERT(opIdx < stats.size(), INVALID_OP_IDX); return stats[opIdx].commTime; }
        size_t GetCommSize(size_t opIdx) { MLSL_ASSERT(opIdx < stats.size(), INVALID_OP_IDX); return stats[opIdx].commSize; }
        unsigned long long GetComputeCycles(size_t opIdx) { MLSL_ASSERT(opIdx < stats.size(), INVALID_OP_IDX); return stats[opIdx].compTime; }
        unsigned long long GetTotalIsolationCommCycles() { return totalIsolationCommTime / (iterations - skip); }
        unsigned long long GetTotalCommCycles() { return totalCommTime; }
        size_t GetTotalCommSize() { return totalCommSize; }
        unsigned long long GetTotalComputeCycles() { return totalCompTime; }
    };

    class SessionImpl : public Session
    {
    private:
        bool isCommited;
        bool isBatchSizeSet;
        PhaseType phaseType;
        size_t globalMinibatchSize;
        vector<OperationImpl*> operations;
        StatisticsImpl* stats;

    public:

        SessionImpl(PhaseType pType) : isCommited(false), isBatchSizeSet(false), phaseType(pType), globalMinibatchSize(0)
        {
            stats = new StatisticsImpl(this);
        }

        ~SessionImpl()
        {
            RemoveOperations();
            delete stats;
        }

        void SetGlobalMinibatchSize(size_t batchSize)
        {
            MLSL_ASSERT(!isBatchSizeSet, "minibatch size can be set only once");
            MLSL_ASSERT(batchSize > 0 && (int)batchSize > 0, "minibatch size must be positive");
            globalMinibatchSize = batchSize;
            isBatchSizeSet = true;
        }

        Statistics* GetStats()
        {
            MLSL_ASSERT((globalMinibatchSize > 0), "Stats cannot be used until globalMinibatchSize is set");
            return stats;
        }
        size_t GetGlobalMinibatchSize() { return globalMinibatchSize; }
        PhaseType GetPhaseType() { return phaseType; }
        OperationRegInfo* CreateOperationRegInfo(OpType opType);
        void DeleteOperationRegInfo(OperationRegInfo* info);
        size_t AddOperation(OperationRegInfo* info, Distribution* dist);
        void RemoveOperations();
        size_t GetOperationCount();
        Operation* GetOperation(size_t idx);

        /* Method needs to be called after all operations are added
         * Session-wide optimization takes place there
         */
        void Commit();
    };

    class OperationImpl : public Operation
    {
    private:
        SessionImpl* session;
        DistributionImpl* dist;

        vector<ActivationImpl*> inActs;
        vector<ActivationImpl*> outActs;
        vector<ParameterSetImpl*> paramSets;
        bool hasParameterSets;
        OpType opType;
        string name;
        int uniqueId;
        size_t localMinibatchSize;
        size_t globalMinibatchOffset;
        size_t opIndex;
        OperationRegInfoImpl* regInfo;

        OperationImpl(const OperationImpl& op);
        OperationImpl& operator=(const OperationImpl& op);

        size_t AllocCommBufs()
        {
            size_t totalSz = 0;

            for (size_t actIdx = 0; actIdx < inActs.size(); actIdx++)
                if (inActs[actIdx]->GetCommBuf())
                    totalSz += inActs[actIdx]->GetCommBuf()->Alloc();

            for (size_t actIdx = 0; actIdx < outActs.size(); actIdx++)
                if (outActs[actIdx]->GetCommBuf())
                    totalSz += outActs[actIdx]->GetCommBuf()->Alloc();

            for (size_t paramIdx = 0; paramIdx < paramSets.size(); paramIdx++)
                if (paramSets[paramIdx]->GetCommBuf())
                    totalSz += paramSets[paramIdx]->GetCommBuf()->Alloc();

            return totalSz;
        }

        void FreeCommBufs()
        {
            for (size_t actIdx = 0; actIdx < inActs.size(); actIdx++)
                if (inActs[actIdx]->GetCommBuf())
                    inActs[actIdx]->GetCommBuf()->Free();

            for (size_t actIdx = 0; actIdx < outActs.size(); actIdx++)
                if (outActs[actIdx]->GetCommBuf())
                    outActs[actIdx]->GetCommBuf()->Free();

            for (size_t paramIdx = 0; paramIdx < paramSets.size(); paramIdx++)
                if (paramSets[paramIdx]->GetCommBuf())
                    paramSets[paramIdx]->GetCommBuf()->Free();
        }

        void Initialize()
        {
            opType = regInfo->GetOpType();
            name = regInfo->GetName();

            for (size_t actIdx = 0; actIdx < regInfo->GetInputCount(); actIdx++)
                inActs.push_back(new ActivationImpl(this, regInfo->GetInput(actIdx), true, actIdx));

            for (size_t actIdx = 0; actIdx < regInfo->GetOutputCount(); actIdx++)
                outActs.push_back(new ActivationImpl(this, regInfo->GetOutput(actIdx), false, actIdx));

            if (regInfo->GetParameterSetCount() > 0) hasParameterSets = true;
            for (size_t paramIdx = 0; paramIdx < regInfo->GetParameterSetCount(); paramIdx++)
                paramSets.push_back(new ParameterSetImpl(this, regInfo->GetParameterSet(paramIdx), paramIdx));

            size_t globalMinibatchSize = session->GetGlobalMinibatchSize();
            size_t dataGroupSize = dist->GetProcessCount(GT_DATA);
            size_t dataParts = dist->GetDataParts();

            MLSL_ASSERT(globalMinibatchSize % dataParts == 0,
                        "global minibatch size (%zu) should be divisible by data partitions (%zu)",
                        globalMinibatchSize, dataParts);

            localMinibatchSize = globalMinibatchSize / dataGroupSize;
            globalMinibatchOffset = localMinibatchSize * dist->GetProcessIdx(GT_DATA);
        }

    public:
        OperationImpl(OperationRegInfo* rInfo, Session* s, Distribution* d, size_t idx)
                      : session(static_cast<SessionImpl*>(s)),
                        dist(static_cast<DistributionImpl*>(d)),
                        hasParameterSets(false),
                        localMinibatchSize(0),
                        globalMinibatchOffset(0),
                        opIndex(idx)
        {
            MLSL_ASSERT(session && rInfo, "session or reg_info is null");
            MLSL_ASSERT((session->GetGlobalMinibatchSize() > 0), "global batch size should be set before operation creation");

            uniqueId = ++opUniqueId;
            regInfo = (OperationRegInfoImpl*)rInfo;
            regInfo->IncrementRefCount();

            if (dist != NULL)
                Initialize();
            else
            {
                opType = regInfo->GetOpType();
                name = regInfo->GetName();
            }
        }

        ~OperationImpl()
        {
            FreeCommBufs();

            for (size_t actIdx = 0; actIdx < inActs.size(); actIdx++)
                delete inActs[actIdx];

            for (size_t actIdx = 0; actIdx < outActs.size(); actIdx++)
                delete outActs[actIdx];

            for (size_t paramIdx = 0; paramIdx < paramSets.size(); paramIdx++)
                delete paramSets[paramIdx];

            inActs.clear();
            outActs.clear();
            paramSets.clear();
            if (regInfo->CanDelete())
                delete regInfo;
        }

        void SetDistribution(Distribution* d)
        {
            MLSL_ASSERT(dist == NULL, "distribution can be set only once");
            MLSL_ASSERT(d, "distribution is NULL");

            dist = static_cast<DistributionImpl*>(d);
            Initialize();
        }

        void SetPrev(Operation* prev, size_t idx, size_t prevOpIdx);
        void SetNext(Operation* next, size_t idx, size_t nextOpIdx);
        int Finalize()
        {
            for (size_t actIdx = 0; actIdx < GetInputCount(); actIdx++)
            {
                MLSL_ASSERT(GetInput(actIdx), "input activation is NULL");
                if (!GetInput(actIdx)->IsPeerSet())
                    GetInput(actIdx)->SetPeer(NULL);
            }

            for (size_t actIdx = 0; actIdx < GetOutputCount(); actIdx++)
            {
                MLSL_ASSERT(GetOutput(actIdx), "output activation is NULL");
                if (!GetOutput(actIdx)->IsPeerSet())
                    GetOutput(actIdx)->SetPeer(NULL);
                GetOutput(actIdx)->InitPeerConnection();
            }

            const size_t BUFSZ = 8192;
            char tbuf[BUFSZ] = "";
            size_t tcou = 0;
            tcou += mysnprintf(tbuf + tcou, BUFSZ, "rank:%zu: operation:%s(%d) - in_acts:%zu out_acts:%zu param_sets:%zu "
                               "local_mb_size:%zu global_mb_off:%zu\n",
                               globalProcessGroup->GetIdx(), name.c_str(), uniqueId, GetInputCount(), GetOutputCount(),
                               GetParameterSetCount(), GetLocalMinibatchSize(), GetGlobalMinibatchOffset());

            for (size_t actIdx = 0; actIdx < GetInputCount(); actIdx++)
            {
                tcou += mysnprintf(tbuf + tcou, BUFSZ - tcou, "INPUT_ACT: %zu: ", actIdx);
                tcou += GetInput(actIdx)->PrintInfo(tbuf + tcou, BUFSZ - tcou);
            }

            for (size_t actIdx = 0; actIdx < GetOutputCount(); actIdx++)
            {
                tcou += mysnprintf(tbuf + tcou, BUFSZ - tcou, "OUTPUT_ACT:%zu: ", actIdx);
                tcou += GetOutput(actIdx)->PrintInfo(tbuf + tcou, BUFSZ - tcou);
            }

            for (size_t paramIdx = 0; paramIdx < GetParameterSetCount(); paramIdx++)
            {
                tcou += mysnprintf(tbuf + tcou, BUFSZ - tcou, "PARAM_SET: %zu: ", paramIdx);
                tcou += GetParameterSet(paramIdx)->PrintInfo(tbuf + tcou, BUFSZ - tcou);
            }

            tbuf[BUFSZ - 1] = 0;
            MLSL_LOG(INFO, "%s", tbuf);

            return 0;
        }
        /* Returns Compute op's distribution related inforation */
        const char* GetName() { return name.c_str(); }
        int GetUniqueId() { return uniqueId; }
        Distribution* GetDistribution() { return dist; }
        Session* GetSession() { return session; }
        OpType GetOpType() { return opType; }
        size_t GetGlobalMinibatchSize() { return session->GetGlobalMinibatchSize(); }
        size_t GetLocalMinibatchSize() { return localMinibatchSize; }
        size_t GetGlobalMinibatchOffset() { return globalMinibatchOffset; }
        size_t GetInputCount() { return inActs.size(); }
        ActivationImpl* GetInput(size_t idx) { return inActs.at(idx); }
        size_t GetOutputCount() { return outActs.size(); }
        ActivationImpl* GetOutput(size_t idx) { return outActs.at(idx); }
        bool HasParameterSets() { return !paramSets.empty(); }
        size_t GetParameterSetCount() { return paramSets.size(); }
        ParameterSetImpl* GetParameterSet(size_t idx) { return paramSets.at(idx); }
        size_t GetOpIndex() { return opIndex; }

        void Commit()
        {
            Finalize();
            AllocCommBufs();
        }
    };
};

#endif /* MLSL_IMPL_HPP */

