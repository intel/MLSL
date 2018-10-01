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
/* Internal DL Specific part of MLSL */

#include <string.h>

#include "common.hpp"
#include "mlsl_impl.hpp"

#ifdef USE_EPLIB
#include "eplib.h"
#endif

namespace MLSL
{
    int opUniqueId;
    pid_t initPid;
    ProcessGroup* globalProcessGroup;
    ProcessGroup* selfProcessGroup;
    QuantParams* globalQuantParam = NULL;


    ActivationImpl::ActivationImpl(OperationImpl* op_, RegActivation* rp, bool isInput_, size_t id_)
                                   : isInput(isInput_), op(op_), actIndex(id_)
    {
        globalFmCount = rp->GetCount();
        dist = static_cast<DistributionImpl*>(op->GetDistribution());
        needComm = false;

        if (isInput == false && op->GetOpType() == OT_CC)
        {
            localFmCount  = globalFmCount;
            globalFmOffset = 0;
            if (dist->GetModelProcessGroup()->GetSize() > 1)
                needReduce = true;
            else
                needReduce = false;
        }
        else
        {
            localFmCount  = globalFmCount / dist->GetModelProcessGroup()->GetSize();
            globalFmOffset = localFmCount * dist->GetModelProcessGroup()->GetIdx();
            needReduce = false;
        }

        featureMapSize = rp->GetSize();
        dataType = rp->GetDataType();
        commBuf = NULL;
        commReq = NULL;
        tmpBufOffset = 0;
        peerAct = NULL;
        isPeerSet = false;
    }

    void OperationImpl::SetPrev(Operation* prev, size_t idx, size_t prevOpIdx)
    {
        MLSL_LOG(INFO, "SetPrev: prev %p, idx %zu, prevOpIdx %zu", prev, idx, prevOpIdx);
        MLSL_ASSERT(CHECK_RANGE(idx, 0, GetInputCount()), "invalid input activation idx");

        ActivationImpl* thisAct = GetInput(idx);
        MLSL_ASSERT(thisAct, "input activation is NULL");

        if (prev == NULL)
        {
            thisAct->SetPeer(NULL);
            return;
        }

        MLSL_ASSERT(GetSession() == prev->GetSession(), "different sessions");
        MLSL_ASSERT(CHECK_RANGE(prevOpIdx, 0, prev->GetOutputCount()), "invalid output activation idx");

        OperationImpl* prevOp = static_cast<OperationImpl*>(prev);
        ActivationImpl* prevAct = prevOp->GetOutput(prevOpIdx);
        MLSL_ASSERT(prevAct, "output activation is NULL");

        prevAct->SetPeer(thisAct);
    }

    void OperationImpl::SetNext(Operation* next, size_t idx, size_t nextOpIdx)
    {
        MLSL_LOG(INFO, "SetNext: next %p, idx %zu, nextOpIdx %zu", next, idx, nextOpIdx);
        MLSL_ASSERT(CHECK_RANGE(idx, 0, GetOutputCount()), "invalid output activation idx");

        ActivationImpl* thisAct = GetOutput(idx);
        MLSL_ASSERT(thisAct, "output activation is NULL");

        if (next == NULL)
        {
            thisAct->SetPeer(NULL);
            return;
        }

        MLSL_ASSERT(GetSession() == next->GetSession(), "different sessions");
        MLSL_ASSERT(CHECK_RANGE(nextOpIdx, 0, next->GetInputCount()), "invalid input activation idx");

        OperationImpl* nextOp = static_cast<OperationImpl*>(next);
        ActivationImpl* nextAct = nextOp->GetInput(nextOpIdx);

        thisAct->SetPeer(nextAct);
    }

    int ActivationImpl::SetPeer(ActivationImpl* act)
    {
        if (act == NULL)
        {
            this->peerAct = NULL;
            this->commBuf = new CommBuf(0);
            isPeerSet = true;
            needComm = false;
            return 0;
        }

        MLSL_ASSERT(act->globalFmCount * act->featureMapSize == globalFmCount * featureMapSize,
                    "prev output activation size must match current input activation size");
        MLSL_ASSERT(isInput != act->isInput, "input-output doesn't pair");
        MLSL_ASSERT(dataType == act->dataType, "datatype must match");
        MLSL_ASSERT(peerAct == NULL || peerAct == act, "peer can be set only once");

        peerAct = act;
        act->peerAct = this;
        isPeerSet = true;
        act->isPeerSet = true;
        return 0;
    }

    int ActivationImpl::InitPeerConnection()
    {
        if (peerAct == NULL) return 0;

        ActivationImpl* act = peerAct;
        ActivationImpl* outAct = isInput ? act : this;
        ActivationImpl* inAct = isInput ? this : act;
        DistributionImpl* outDist = outAct->dist;
        DistributionImpl* inDist = inAct->dist;

        if (globalProcessGroup->GetSize() > 1 && (outAct->needReduce || outDist != inDist))
        {
            outAct->needComm = true;
            inAct->needComm = true;
        }

        if (needComm == true)
        {
            outAct->commReq = CommCreateRequest(outAct->dataType, outAct->op->GetUniqueId(), CommDesc::FPROP);
            inAct->commReq = CommCreateRequest(outAct->dataType, inAct->op->GetUniqueId(), CommDesc::BPROP);
            if (outAct->needReduce && outAct->dist == inAct->dist)
            {
                MLSL_LOG(DEBUG, "case 1");
                MLSL_LOG(DEBUG, "AddReduceScatter - %zu * %zu * %zu = %zu",
                         inAct->localFmCount, op->GetLocalMinibatchSize(), inAct->featureMapSize,
                         inAct->localFmCount * op->GetLocalMinibatchSize() * inAct->featureMapSize);
                outAct->commReq->GetDesc()->AddReduceScatter(inAct->localFmCount * op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                             inDist->GetModelProcessGroup());
                outAct->commReq->Setup();
                outAct->BIPackReduceScatter();
                inAct->BIUnpackReduceScatter();
                inAct->commReq->GetDesc()->AddAllGather(inAct->localFmCount * op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                        inDist->GetModelProcessGroup());
                inAct->commReq->Setup();
                inAct->BIPackAllGather();
                outAct->BIUnpackAllGather();
            }
            else if (outAct->needReduce && inAct->dist->GetModelProcessGroup()->GetSize() == 1 &&
                     outDist->GetDataProcessGroup()->GetSize() == inDist->GetDataProcessGroup()->GetSize())
            {
                MLSL_LOG(DEBUG, "case 2");
                outAct->commReq->GetDesc()->AddAllReduce(outAct->localFmCount * outAct->op->GetLocalMinibatchSize() * outAct->featureMapSize,
                                                         outDist->GetModelProcessGroup());
                outAct->BIPackAllReduce();
                inAct->BIUnpackAllReduce();
                outAct->commReq->Setup();
                inAct->commReq->Setup(); // No comm needed
            }
            else if (outAct->needReduce && inDist->GetModelProcessGroup()->GetSize() == 1 &&
                     inDist->GetDataProcessGroup()->GetSize() % outDist->GetDataProcessGroup()->GetSize() == 0 &&
                     inDist->GetDataProcessGroup()->GetSize() == outDist->GetModelProcessGroup()->GetSize() * outDist->GetDataProcessGroup()->GetSize())
            {
                MLSL_LOG(DEBUG, "case 3");
                outAct->commReq->GetDesc()->AddReduceScatter(inAct->localFmCount * inAct->op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                             outDist->GetModelProcessGroup());
                outAct->commReq->Setup();
                outAct->BIPackReduceScatter2();
                inAct->BIUnpackReduceScatter();
                inAct->commReq->GetDesc()->AddAllGather(inAct->localFmCount * inAct->op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                        outDist->GetModelProcessGroup());
                inAct->commReq->Setup();
                inAct->BIPackAllGather();
                outAct->BIUnpackAllGather2();
            }
            else if (outAct->needReduce == false && outAct->dist->GetModelProcessGroup()->GetSize() == 1)
            {
                MLSL_LOG(DEBUG, "case 4");
                outAct->commReq->GetDesc()->AddAlltoAll(inAct->localFmCount * outAct->op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                        inDist->GetModelProcessGroup());
                outAct->commReq->Setup();
                outAct->BIBuildAlltoAll(inAct);
                inAct->commReq->GetDesc()->AddAlltoAll(inAct->localFmCount * outAct->op->GetLocalMinibatchSize() * inAct->featureMapSize,
                                                       inDist->GetModelProcessGroup());
                inAct->commReq->Setup();
                inAct->BIBuildAlltoAll(outAct);
            }
            else if (outAct->needReduce == false && inAct->dist->GetModelProcessGroup()->GetSize() == 1)
            {
                MLSL_LOG(DEBUG, "case 5");
                outAct->commReq->GetDesc()->AddAlltoAll(outAct->localFmCount * inAct->op->GetLocalMinibatchSize() * outAct->featureMapSize,
                                                        outDist->GetModelProcessGroup());
                outAct->commReq->Setup();
                outAct->BIBuildAlltoAll(inAct);
                inAct->commReq->GetDesc()->AddAlltoAll(outAct->localFmCount * inAct->op->GetLocalMinibatchSize() * outAct->featureMapSize,
                                                       outDist->GetModelProcessGroup());
                inAct->commReq->Setup();
                inAct->BIBuildAlltoAll(outAct);
            }
            else
                MLSL_ASSERT(0, "this case is not supported yet");
        }

        if (inAct->commReq != NULL)
            inAct->commBuf = new CommBuf(inAct->commReq->GetBufSize());
        else
            inAct->commBuf = new CommBuf(0);
        if (outAct->commReq != NULL)
            outAct->commBuf = new CommBuf(outAct->commReq->GetBufSize());
        else
            outAct->commBuf = new CommBuf(0);

        return 0;
    }

    void ActivationImpl::BIPackReduceScatter()
    {
        size_t modelParts = dist->GetModelProcessGroup()->GetSize();
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount / modelParts;
        for (size_t i = 0; i < modelParts; i++)
            packBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, i * fmCount, fmCount, featureMapSize,
                                                       dataType, i * localMbSize * fmCount * featureMapSize));
        size_t dataTypeSize = (dataType == DT_FLOAT) ? 4 : 8;
        tmpBufOffset = modelParts * localMbSize * fmCount * featureMapSize * dataTypeSize;
    }
    void ActivationImpl::BIPackReduceScatter2()
    {
        size_t modelParts = dist->GetModelProcessGroup()->GetSize();
        size_t localMbSize = op->GetLocalMinibatchSize() / modelParts;
        size_t fmCount = localFmCount;
        for (size_t i = 0; i < modelParts; i++)
            packBlocks.push_back(new CommBlockInfoImpl(i * localMbSize, localMbSize, 0, fmCount, featureMapSize,
                                                       dataType, i * localMbSize * fmCount * featureMapSize));
        size_t dataTypeSize = (dataType == DT_FLOAT) ? 4 : 8;
        tmpBufOffset = modelParts * localMbSize * fmCount * featureMapSize * dataTypeSize;
    }
    void ActivationImpl::BIUnpackReduceScatter()
    {
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount;
        size_t offset = 0;
        unpackBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, 0, fmCount, featureMapSize, dataType, offset));
    }
    void ActivationImpl::BIPackAllReduce()
    {
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount;
        packBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, 0, fmCount, featureMapSize, dataType, 0));
        size_t dataTypeSize = (dataType == DT_FLOAT) ? 4 : 8;
        tmpBufOffset = localMbSize * fmCount * featureMapSize * dataTypeSize;
    }
    void ActivationImpl::BIUnpackAllReduce()
    {
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount;
        unpackBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, 0, fmCount, featureMapSize, dataType, 0));
    }
    void ActivationImpl::BIPackAllGather()
    {
        size_t fmIdx = dist->GetModelProcessGroup()->GetIdx();
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount;
        packBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, 0, fmCount, featureMapSize,
                                                   dataType, fmIdx * localMbSize * fmCount * featureMapSize));
    }
    void ActivationImpl::BIUnpackAllGather()
    {
        size_t modelParts = dist->GetModelProcessGroup()->GetSize();
        size_t localMbSize = op->GetLocalMinibatchSize();
        size_t fmCount = localFmCount / modelParts;
        for (size_t i = 0; i < modelParts; i++)
            unpackBlocks.push_back(new CommBlockInfoImpl(0, localMbSize, i * fmCount, fmCount, featureMapSize,
                                                         dataType, i * localMbSize * fmCount * featureMapSize));
    }
    void ActivationImpl::BIUnpackAllGather2()
    {
        size_t modelParts = dist->GetModelProcessGroup()->GetSize();
        size_t localMbSize = op->GetLocalMinibatchSize() / modelParts;
        size_t fmCount = localFmCount;
        for (size_t i = 0; i < modelParts; i++)
            unpackBlocks.push_back(new CommBlockInfoImpl(i * localMbSize, localMbSize, 0, fmCount, featureMapSize,
                                                        dataType, i * localMbSize * fmCount * featureMapSize));
    }

    void ActivationImpl::BIBuildAlltoAll(ActivationImpl* inAct)
    {
        ActivationImpl* outAct = this;
        MLSL_ASSERT(outAct->dist->GetModelProcessGroup()->GetSize() == 1 || inAct->dist->GetModelProcessGroup()->GetSize() == 1,
                    "one of ModelGroupSize should be 1");
        ProcessGroup* modelProcessGroup = outAct->dist->GetModelProcessGroup()->GetSize() == 1 ? inAct->dist->GetModelProcessGroup() : outAct->dist->GetModelProcessGroup();
        size_t localMbSize = MIN(outAct->op->GetLocalMinibatchSize(), inAct->op->GetLocalMinibatchSize());
        size_t fmCountxSize = MIN(outAct->localFmCount * outAct->featureMapSize, inAct->localFmCount * inAct->featureMapSize);
        size_t outFmCount = fmCountxSize / outAct->featureMapSize;
        size_t inFmCount = fmCountxSize / inAct->featureMapSize;
        size_t blockIdx = 0;
        for (size_t i = 0; i < outAct->op->GetLocalMinibatchSize(); i += localMbSize)
            for (size_t j = 0; j < outAct->localFmCount; j += outFmCount)
            {
                outAct->packBlocks.push_back(new CommBlockInfoImpl(i, localMbSize, j, outFmCount, outAct->featureMapSize,
                                                                  dataType, blockIdx * localMbSize * fmCountxSize));
                blockIdx++;
            }

        MLSL_ASSERT(blockIdx == modelProcessGroup->GetSize(), "blockIdx(%zu) should be equal to ProcessGroupSize(%zu)",
                    blockIdx, modelProcessGroup->GetSize());
        blockIdx = 0;
        for (size_t i = 0; i < inAct->op->GetLocalMinibatchSize(); i += localMbSize)
            for (size_t j = 0; j < inAct->localFmCount; j += inFmCount)
            {
                inAct->unpackBlocks.push_back(new CommBlockInfoImpl(i, localMbSize, j, inFmCount, inAct->featureMapSize,
                                                                    dataType, blockIdx * localMbSize * fmCountxSize));
                blockIdx++;
            }

        MLSL_ASSERT(blockIdx == modelProcessGroup->GetSize(), "blockIdx(%zu) should be equal to ProcessGroupSize(%zu)",
                    blockIdx, modelProcessGroup->GetSize());
        size_t dataTypeSize = (dataType == DT_FLOAT) ? 4 : 8;
        tmpBufOffset = modelProcessGroup->GetSize() * localMbSize * fmCountxSize * dataTypeSize;
    }

    void ActivationImpl::StartComm(void* buf)
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        actIndex,        /* entIdx */
                        true,            /* isCompTime */
                        false,           /* isParam */
                        isInput,         /* isInputOrIncrement */
                        StatEvent::Start /* actionType */ );
        stats->UpdateStats(event);

        if (needComm) commReq->Start(buf, (char*)buf + tmpBufOffset);

        event.isCompTime = false;
        stats->UpdateStats(event);
    }

    void* ActivationImpl::WaitComm()
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        actIndex,       /* entIdx */
                        true,           /* isCompTime */
                        false,          /* isParam */
                        isInput,        /* isInputOrIncrement */
                        StatEvent::Wait /* actionType */ );
        stats->UpdateStats(event);

        void* ptr = NULL;
        if (needComm)
            if (peerAct != NULL && peerAct->commReq != NULL)
                ptr = peerAct->commReq->Wait();

        event.isCompTime = false;
        stats->UpdateStats(event);

        return ptr;
    }

    ParameterSetImpl::ParameterSetImpl(OperationImpl* oi, RegParameterSet* rp, size_t id_)
                                       : distributedUpdate(rp->GetDistributedUpdate()),
                                         op(oi),
                                         dist(static_cast<DistributionImpl*>(oi->GetDistribution())),
                                         paramIndex(id_)
    {
        globalKernelCount = rp->GetCount();
        localKernelCount  = globalKernelCount / dist->GetModelProcessGroup()->GetSize();
        globalKernelOffset = localKernelCount * dist->GetModelProcessGroup()->GetIdx();

        needComm = false;
        if (dist->GetDataProcessGroup()->GetSize() > 1) needComm = true;

        if (distributedUpdate == true)
        {
            ownedKernelCount = (localKernelCount + dist->GetDataProcessGroup()->GetSize() - 1) / dist->GetDataProcessGroup()->GetSize();
            localKernelCount = ownedKernelCount * dist->GetDataProcessGroup()->GetSize();
            ownedKernelOffset = ownedKernelCount * dist->GetDataProcessGroup()->GetIdx();
        }
        else
        {
            ownedKernelCount = localKernelCount;
            ownedKernelOffset = 0;
        }

        kernelSize = rp->GetSize();
        dataType = rp->GetDataType();
        size_t tmpBufSize = 0;
        commBuf = NULL;
        if (needComm)
        {
            gradReq = CommCreateRequest(dataType, op->GetUniqueId(), CommDesc::PARAM_GRAD);
            gradReq->SetCompressionType(rp->GetCompressionType());
            if (distributedUpdate == true)
                gradReq->GetDesc()->AddReduceScatter(ownedKernelCount * kernelSize, dist->GetDataProcessGroup());
            else
                gradReq->GetDesc()->AddAllReduce(ownedKernelCount * kernelSize, dist->GetDataProcessGroup());
            gradReq->Setup();
            tmpBufSize = gradReq->GetBufSize();
            if (distributedUpdate == true)
            {
                incReq = CommCreateRequest(dataType, op->GetUniqueId(), CommDesc::PARAM_INC);
                incReq->GetDesc()->AddAllGather(ownedKernelCount * kernelSize, dist->GetDataProcessGroup());
                incReq->Setup();
                tmpBufSize = MAX(tmpBufSize, incReq->GetBufSize());
            }
            else
                incReq = NULL;
        }
        else
        {
            gradReq = NULL;
            incReq = NULL;
        }
        commBuf = new CommBuf(tmpBufSize);
        compType = rp->GetCompressionType();
    }

    void ParameterSetImpl::StartGradientComm(void* buf)
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        paramIndex,      /* entIdx */
                        true,            /* isCompTime */
                        true,            /* isParam */
                        false,           /* isInputOrIncrement */
                        StatEvent::Start /* actionType */ );
        stats->UpdateStats(event);

        if (needComm) gradReq->Start(buf, (distributedUpdate == true) ? commBuf->GetPtr() : buf);

        event.isCompTime = false;
        stats->UpdateStats(event);
    }

    void* ParameterSetImpl::WaitGradientComm()
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        paramIndex,     /* entIdx */
                        true,           /* isCompTime */
                        true,           /* isParam */
                        false,          /* isInputOrIncrement */
                        StatEvent::Wait /* actionType */ );
        stats->UpdateStats(event);

        void* ptr = NULL;
        if (needComm) ptr = gradReq->Wait();

        event.isCompTime = false;
        stats->UpdateStats(event);

        return ptr;
    }

    void* ParameterSetImpl::TestGradientComm(bool* isCompleted)
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        paramIndex,     /* entIdx */
                        true,           /* isCompTime */
                        true,           /* isParam */
                        false,          /* isInputOrIncrement */
                        StatEvent::Test /* actionType */ );
        stats->UpdateStats(event);

        void* ptr = NULL;
        if (needComm) ptr = gradReq->Test(isCompleted, false);
        else *isCompleted = true;

        event.isCompTime = false;
        stats->UpdateStats(event);

        return ptr;
    }

    void ParameterSetImpl::StartIncrementComm(void* buf)
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        paramIndex,      /* entIdx */
                        true,            /* isCompTime */
                        true,            /* isParam */
                        true,            /* isInputOrIncrement */
                        StatEvent::Start /* actionType */ );
        stats->UpdateStats(event);

        if (needComm && distributedUpdate) incReq->Start(buf, buf);

        event.isCompTime = false;
        stats->UpdateStats(event);
    }

    void* ParameterSetImpl::WaitIncrementComm()
    {
        StatisticsImpl* stats = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        StatEvent event(op->GetOpIndex(),
                        paramIndex,     /* entIdx */
                        true,           /* isCompTime */
                        true,           /* isParam */
                        true,           /* isInputOrIncrement */
                        StatEvent::Wait /* actionType */ );
        stats->UpdateStats(event);

        void* ptr = NULL;
        if (needComm && distributedUpdate) ptr = incReq->Wait();

        event.isCompTime = false;
        stats->UpdateStats(event);

        return ptr;
    }

    OperationRegInfo* SessionImpl::CreateOperationRegInfo(OpType opType)
    {
        return new OperationRegInfoImpl(opType);
    }

    void SessionImpl::DeleteOperationRegInfo(OperationRegInfo* info)
    {
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(info);
        if (p->CanDelete())
            delete p;
    }

    size_t SessionImpl::AddOperation(OperationRegInfo* info, Distribution* dist)
    {
        OperationImpl* oi = new OperationImpl(info, this, dist, operations.size() /* op_idx */);
        operations.push_back(oi);
        return operations.size() - 1;
    }

    void SessionImpl::RemoveOperations()
    {
        for (size_t opIdx = 0; opIdx < operations.size(); opIdx++)
            delete operations.at(opIdx);
        operations.clear();
    }

    void SessionImpl::Commit()
    {
        MLSL_ASSERT(!isCommited, "commit should be called only once");
        MLSL_ASSERT(globalMinibatchSize > 0, "set global mini-batch size before commit");
        stats->Initialize();

        for (size_t opIdx = 0; opIdx < operations.size(); opIdx++)
            operations.at(opIdx)->Commit();

        stats->CollectIsolationStats();
        isCommited = true;
    }

    size_t SessionImpl::GetOperationCount()
    {
        return operations.size();
    }

    Operation* SessionImpl::GetOperation(size_t idx)
    {
        return operations.at(idx);
    }

    void DistributionImpl::Barrier(GroupType gt)
    {
        CommRequest* req = CommCreateRequest(DataType(-1), -1, CommDesc::GENERIC);
        ProcessGroup* group = GetGroupByType(gt);
        req->GetDesc()->AddBarrier(group);
        req->Setup();
        req->Start(NULL, NULL);
        delete req;
    }

    CommReq* DistributionImpl::Bcast(void* buffer, size_t count, DataType dataType, size_t rootIdx, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddBcast(count, GetGroupByType(gt), rootIdx);
        req->Setup();
        req->Start(buffer, NULL);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::Reduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, size_t rootIdx, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddReduce(count, GetGroupByType(gt), rootIdx, GetReduceOpByType(rt));
        req->Setup();
        req->Start(sendBuffer, recvbuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::AllReduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddAllReduce(count, GetGroupByType(gt), GetReduceOpByType(rt));
        req->Setup();
        req->Start(sendBuffer, recvbuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::AlltoAll(void* sendBuffer, size_t sendCount, void* recvBuffer,  DataType dataType, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddAlltoAll(sendCount, GetGroupByType(gt));
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::AlltoAllv(void* sendBuffer, size_t* sendCounts, size_t* sendOffsets, void* recvBuffer, size_t* recvCounts, size_t* recvOffsets, DataType dataType, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddAlltoAllv(0, GetGroupByType(gt), sendCounts, sendOffsets, recvCounts, recvOffsets);
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::Gather(void* sendBuffer, size_t sendCount, void* recvBuffer,  DataType dataType, size_t rootIdx, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddGather(sendCount, GetGroupByType(gt), rootIdx);
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::AllGather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddAllGather(sendCount, GetGroupByType(gt));
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::AllGatherv(void* sendBuffer, size_t sendCount, void* recvBuffer, size_t* recvCounts, DataType dataType, GroupType gt) 
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddAllGatherv(sendCount, GetGroupByType(gt), recvCounts);
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::Scatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, size_t rootIdx, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddScatter(recvCount, GetGroupByType(gt), rootIdx);
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }

    CommReq* DistributionImpl::ReduceScatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, ReductionType rt, GroupType gt)
    {
        CommRequest* req = CommCreateRequest(dataType, -1, CommDesc::GENERIC);
        req->GetDesc()->AddReduceScatter(recvCount, GetGroupByType(gt), GetReduceOpByType(rt));
        req->SetInPlace((sendBuffer == recvBuffer) ? true : false);
        req->Setup();
        req->Start(sendBuffer, recvBuffer);
        RequestStorage::GetObject().RegisterRequest(req);
        return (CommReq*)req;
    }
};

