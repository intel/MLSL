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
/* Public External Interface to MLSL */

#include "comm.hpp"
#include "mlsl_impl.hpp"
#include "version.hpp"
#include "sysinfo.hpp"

#define MLSL_CONFIG_COLOR  "color"
#define MLSL_CONFIG_DELIM  ","
#define MLSL_CONFIG_ASSIGN "="

namespace MLSL
{
    static bool isInitialized = false;
    static int color = 0;
    static bool isGlobalGroupChanged = false;

    size_t Distribution::GetProcessCount(GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);

        if (gt == GT_DATA)
            return p->GetDataProcessGroup()->GetSize();

        if (gt == GT_MODEL)
            return p->GetModelProcessGroup()->GetSize();

        return Environment::GetEnv().GetProcessCount();
    }

    size_t Distribution::GetProcessIdx(GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);

        if (gt == GT_DATA)
            return p->GetDataProcessGroup()->GetIdx();

        if (gt == GT_MODEL)
            return p->GetModelProcessGroup()->GetIdx();

        return Environment::GetEnv().GetProcessIdx();
    }

    void Distribution::Barrier(GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        p->Barrier(gt);
    }

    CommReq* Distribution::Bcast(void* buffer, size_t count, DataType dataType, size_t rootIdx, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->Bcast(buffer, count, dataType, rootIdx, gt);
    }

    CommReq* Distribution::Reduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, size_t rootIdx, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->Reduce(sendBuffer, recvbuffer, count, dataType, rt, rootIdx, gt);
    }

    CommReq* Distribution::AllReduce(void* sendBuffer, void* recvbuffer, size_t count, DataType dataType, ReductionType rt, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->AllReduce(sendBuffer, recvbuffer, count, dataType, rt, gt);
    }

    CommReq* Distribution::AlltoAll(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->AlltoAll(sendBuffer, sendCount, recvBuffer, dataType, gt);
    }

    CommReq* Distribution::AlltoAllv(void* sendBuffer, size_t* sendCounts, size_t* sendOffsets, void* recvBuffer, size_t* recvCounts, size_t* recvOffsets, DataType dataType, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->AlltoAllv(sendBuffer, sendCounts, sendOffsets, recvBuffer, recvCounts, recvOffsets, dataType, gt);
    }

    CommReq* Distribution::Gather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, size_t rootIdx, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->Gather(sendBuffer, sendCount, recvBuffer, dataType, rootIdx, gt);
    }

    CommReq* Distribution::AllGather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->AllGather(sendBuffer, sendCount, recvBuffer, dataType, gt);
    }

    CommReq* Distribution::AllGatherv(void* sendBuffer, size_t sendCount, void* recvBuffer, size_t* recvCounts, DataType dataType, GroupType gt) 
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->AllGatherv(sendBuffer, sendCount, recvBuffer, recvCounts, dataType, gt);
    }

    CommReq* Distribution::Scatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, size_t rootIdx, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->Scatter(sendBuffer, recvBuffer, recvCount, dataType, rootIdx, gt);
    }

    CommReq* Distribution::ReduceScatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, ReductionType rt, GroupType gt)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(this);
        return p->ReduceScatter(sendBuffer, recvBuffer, recvCount, dataType, rt, gt);
    }

    void OperationRegInfo::SetName(const char* name)
    {
        MLSL_ASSERT(name, "name is null");
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(this);
        p->SetName(name);
    }

    size_t OperationRegInfo::AddInput(size_t featureMapCount, size_t featureMapSize, DataType dataType)
    {
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(this);
        MLSL_ASSERT(featureMapCount > 0 && featureMapSize > 0 && (int)featureMapCount > 0 && (int)featureMapSize > 0,
                    "count and size should be positive\n");
        return p->AddInput(new RegActivation(featureMapCount, featureMapSize, dataType));
    }

    size_t OperationRegInfo::AddOutput(size_t featureMapCount, size_t featureMapSize, DataType dataType)
    {
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(this);
        MLSL_ASSERT(featureMapCount > 0 && featureMapSize > 0 && (int)featureMapCount > 0 && (int)featureMapSize > 0,
                    "count and size should be positive\n");
        return p->AddOutput(new RegActivation(featureMapCount, featureMapSize, dataType));
    }

    size_t OperationRegInfo::AddParameterSet(size_t kernelCount, size_t kernelSize, DataType dataType, bool distributedUpdate, CompressionType compressType)
    {
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(this);
        MLSL_ASSERT(kernelCount > 0 && kernelSize > 0 && (int)kernelCount > 0 && (int)kernelSize > 0,
                    "count and size should be positive\n");
        if (compressType != CT_NONE)
            MLSL_ASSERT(globalQuantParam, "For use quantization you should set quantization parameters");
        return p->AddParameterSet(new RegParameterSet(kernelCount, kernelSize, dataType, distributedUpdate, compressType));
    }

    void OperationRegInfo::Validate(Distribution* dist)
    {
        OperationRegInfoImpl* p = static_cast<OperationRegInfoImpl*>(this);
        p->Validate(dist);
    }

    void Operation::SetPrev(Operation* prev, size_t idx, size_t prevOpIdx)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        p->SetPrev(prev, idx, prevOpIdx);
    }

    void Operation::SetNext(Operation* next, size_t idx, size_t nextOpIdx)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        p->SetNext(next, idx, nextOpIdx);
    }

    const char* Operation::GetName()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetName();
    }

    Distribution* Operation::GetDistribution()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetDistribution();
    }

    Session* Operation::GetSession()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetSession();
    }

    OpType Operation::GetOpType()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetOpType();
    }

    size_t Operation::GetGlobalMinibatchSize()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetGlobalMinibatchSize();
    }

    size_t Operation::GetLocalMinibatchSize()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetLocalMinibatchSize();
    }

    size_t Operation::GetGlobalMinibatchOffset()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetGlobalMinibatchOffset();
    }

    size_t Operation::GetInputCount()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetInputCount();
    }

    Activation* Operation::GetInput(size_t idx)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetInput(idx);
    }

    size_t Operation::GetOutputCount()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetOutputCount();
    }

    Activation* Operation::GetOutput(size_t idx)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetOutput(idx);
    }

    bool Operation::HasParameterSets()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->HasParameterSets();
    }

    size_t Operation::GetParameterSetCount()
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetParameterSetCount();
    }

    void Operation::SetDistribution(Distribution* dist)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->SetDistribution(dist);
    }

    ParameterSet* Operation::GetParameterSet(size_t idx)
    {
        OperationImpl* p = static_cast<OperationImpl*>(this);
        return p->GetParameterSet(idx);
    }

    size_t Activation::GetGlobalFmCount()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetGlobalFmCount();
    }

    size_t Activation::GetGlobalFmOffset()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetGlobalFmOffset();
    }

    size_t Activation::GetLocalFmCount()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetLocalFmCount();
    }

    size_t Activation::GetPackBlockCount()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetPackBlockCount();
    }

    size_t Activation::GetUnpackBlockCount()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetUnpackBlockCount();
    }

    CommBlockInfo* Activation::GetPackBlock(size_t blockIdx)
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetPackBlock(blockIdx);
    }

    CommBlockInfo* Activation::GetUnpackBlock(size_t blockIdx)
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetUnpackBlock(blockIdx);
    }

    DataType Activation::GetDataType()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetDataType();
    }

    size_t Activation::GetFmSize()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetFmSize();
    }

    void* Activation::GetCommBuf()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetCommBuf()->GetPtr();
    }

    size_t Activation::GetCommBufSize()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->GetCommBuf()->GetSize();
    }

    void Activation::StartComm(void* buf)
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        p->StartComm(buf);
    }

    void* Activation::WaitComm()
    {
        ActivationImpl* p = static_cast<ActivationImpl*>(this);
        return p->WaitComm();
    }

    size_t ParameterSet::GetGlobalKernelCount()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetGlobalKernelCount();
    }

    size_t ParameterSet::GetGlobalKernelOffset()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetGlobalKernelOffset();
    }

    size_t ParameterSet::GetLocalKernelCount()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetLocalKernelCount();
    }

    size_t ParameterSet::GetOwnedKernelCount()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetOwnedKernelCount();
    }

    size_t ParameterSet::GetOwnedKernelOffset()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetOwnedKernelOffset();
    }

    DataType ParameterSet::GetDataType()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetDataType();
    }

    size_t ParameterSet::GetKernelSize()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->GetKernelSize();
    }

    bool ParameterSet::IsDistributedUpdate()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->IsDistributedUpdate();
    }

    void ParameterSet::StartGradientComm(void* buf)
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        p->StartGradientComm(buf);
    }

    void* ParameterSet::WaitGradientComm()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->WaitGradientComm();
    }

    void* ParameterSet::TestGradientComm(bool* isCompleted)
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->TestGradientComm(isCompleted);
    }

    void ParameterSet::StartIncrementComm(void* buf)
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        p->StartIncrementComm(buf);
    }

    void* ParameterSet::WaitIncrementComm()
    {
        ParameterSetImpl* p = static_cast<ParameterSetImpl*>(this);
        return p->WaitIncrementComm();
    }

    size_t CommBlockInfo::GetMbOffset()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetMbOffset();
    }

    size_t CommBlockInfo::GetMbCount()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetMbCount();
    }

    size_t CommBlockInfo::GetFmOffset()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetFmOffset();
    }

    size_t CommBlockInfo::GetFmCount()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetFmCount();
    }

    size_t CommBlockInfo::GetFmSize()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetFmSize();
    }

    DataType CommBlockInfo::GetDataType()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetDataType();
    }

    size_t CommBlockInfo::GetBufOffset()
    {
        CommBlockInfoImpl* p = static_cast<CommBlockInfoImpl*>(this);
        return p->GetBufOffset();
    }

    void Statistics::Stop()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->Stop();
    }

    void Statistics::Print()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->Print();
    }

    void Statistics::Start()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->Start();
    }

    void Statistics::Reset()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->Reset();
    }

    bool Statistics::IsStarted()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->IsStarted();
    }

    bool Statistics::IsEnabled()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->IsEnabled();
    }

    unsigned long long Statistics::GetIsolationCommCycles(size_t opIdx)
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetIsolationCommCycles(opIdx);
    }

    unsigned long long Statistics::GetCommCycles(size_t opIdx)
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetCommCycles(opIdx);
    }

    size_t Statistics::GetCommSize(size_t opIdx)
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetCommSize(opIdx);
    }

    unsigned long long Statistics::GetComputeCycles(size_t opIdx)
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetComputeCycles(opIdx);
    }

    unsigned long long Statistics::GetTotalIsolationCommCycles()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetTotalIsolationCommCycles();
    }

    unsigned long long Statistics::GetTotalCommCycles()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetTotalCommCycles();
    }

    size_t Statistics::GetTotalCommSize()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetTotalCommSize();
    }

    unsigned long long Statistics::GetTotalComputeCycles()
    {
        StatisticsImpl* p = static_cast<StatisticsImpl*>(this);
        return p->GetTotalComputeCycles();
    }

    void Session::SetGlobalMinibatchSize(size_t globalMinibatchSize)
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        p->SetGlobalMinibatchSize(globalMinibatchSize);
    }

    size_t Session::GetGlobalMinibatchSize()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->GetGlobalMinibatchSize();
    }

    PhaseType Session::GetPhaseType()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->GetPhaseType();
    }

    OperationRegInfo* Session::CreateOperationRegInfo(OpType opType)
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->CreateOperationRegInfo(opType);
    }

    void Session::DeleteOperationRegInfo(OperationRegInfo* info)
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->DeleteOperationRegInfo(info);
    }

    size_t Session::AddOperation(OperationRegInfo* info, Distribution* dist)
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->AddOperation(info, dist);
    }

    void Session::RemoveOperations()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->RemoveOperations();
    }

    size_t Session::GetOperationCount()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->GetOperationCount();
    }

    Operation* Session::GetOperation(size_t idx)
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->GetOperation(idx);
    }

    void Session::Commit()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        p->Commit();
    }

    Statistics* Session::GetStats()
    {
        SessionImpl* p = static_cast<SessionImpl*>(this);
        return p->GetStats();
    }

    Environment& Environment::GetEnv()
    {
        static Environment env;
        return env;
    }

    void Environment::Configure(const char* config)
    {
        /* TBD: Good-looking parsing for config
         * Also, all configuration details (color etc.)
         * should be moved to some object
        */
        if (config == NULL) return;

        char* arg = strdup(config);
        char* token;
        while ((token = strsep(&arg, MLSL_CONFIG_ASSIGN)))
        {
            if (!strcmp(token, MLSL_CONFIG_COLOR))
            {
                color = atoi(strsep(&arg, MLSL_CONFIG_DELIM));
                /* TBD: Need to decide whether we are going to call Configure before Init
                 * or after. The first option should be preferable as allows for selecting
                 * backends other than MPI. This means logic for setting color needs to be
                 * moved to some other API function e.g. AddSettings()
                */
                globalProcessGroup = SplitProcessGroup(globalProcessGroup, color);
                SetGlobalProcessGroup(globalProcessGroup);
                isGlobalGroupChanged = true;
                break;
            }
        }
    }

    void AutoConfig(AutoConfigType autoConfigType)
    {
        if (autoConfigType == NO_AUTO_CONFIG) return;

        SysInfo sysInfo;
        sysInfo.Initialize();
        CpuInfo cpuInfo = sysInfo.GetCpuInfo();
        NetInfo netInfo = sysInfo.GetNetInfo();

        char* impi_fabrics = getenv("I_MPI_FABRICS");

        switch (autoConfigType)
        {
            case NET_AUTO_CONFIG:
                MLSL_LOG(DEBUG, "net type: %s", ConvertToNetName(netInfo.netType).c_str());
                if (netInfo.netType == ETH ||
                    (impi_fabrics && ((strcmp(impi_fabrics, "tcp") == 0) || (strcmp(impi_fabrics, "shm:tcp") == 0))))
                {
                    setenv("MLSL_LARGE_MSG_CHUNKS", "128", 0);
                }
                break;
            case CPU_AUTO_CONFIG:
                MLSL_LOG(DEBUG, "cpu type: %s, cores: %zu, threads: %zu",
                         ConvertToCpuName(cpuInfo.cpuType).c_str(), cpuInfo.cpuCoresCount, cpuInfo.cpuThreadsCount);
                break;
            case NET_AND_CPU_AUTO_CONFIG:
                MLSL_LOG(DEBUG, "net type: %s", ConvertToNetName(netInfo.netType).c_str());
                MLSL_LOG(DEBUG, "cpu type: %s, cores: %zu, threads: %zu",
                         ConvertToCpuName(cpuInfo.cpuType).c_str(), cpuInfo.cpuCoresCount, cpuInfo.cpuThreadsCount);
                break;
            default:
                MLSL_ASSERT(0, "invalid MLSL_AUTO_CONFIG_TYPE: %d", autoConfigType);
        }
    }

    void Environment::Init(int* argc, char** argv[])
    {
        MLSL_ASSERT(!isInitialized, "Intel(R) MLSL can be initialized only once");

        initPid = getpid();
        MLSL_LOG(DEBUG, "init_pid %d", initPid);

        ParseEnvVars();

        AutoConfigType autoConfigType = (AutoConfigType)envData.autoConfigType;
        AutoConfig(autoConfigType);

        opUniqueId = -1;
        CommInit(argc, argv);
        globalProcessGroup = GetGlobalProcessGroup();
        selfProcessGroup  = GetSelfProcessGroup();

        if (globalProcessGroup->GetIdx() == 0)
        {
            PrintEnvVars();
            MLSL_LOG(INFO, MLSL_PACKAGE_VERSION);
            MLSL_LOG(INFO, "Intel(R) MLSL API: %d.%d", MLSL_MAJOR_VERSION, MLSL_MINOR_VERSION);
            MLSL_LOG(DEBUG, "git version: %s", MLSL_GIT_VERSION);
        }

        isInitialized = true;
    }

    void Environment::Finalize()
    {
        if (!isInitialized)
        {
            MLSL_LOG(INFO, "Intel(R) MLSL isn't initialized, skip finalization");
            return;
        }

        if (initPid != getpid())
        {
            MLSL_LOG(INFO, "different pids: init_pid %d, current_pid %d, skip finalization", initPid, getpid());
            return;
        }

        if (isGlobalGroupChanged) FreeProcessGroup(globalProcessGroup);
        CommFinalize();
        globalProcessGroup = NULL;
        selfProcessGroup   = NULL;
        opUniqueId = -1;

        size_t incompleteReqCount = RequestStorage::GetObject().GetSize();
        if (incompleteReqCount) MLSL_LOG(INFO, "there are %zu incompleted requests", incompleteReqCount);

        if (globalQuantParam != NULL)
#ifdef USE_EPLIB
            EPLIB_free(globalQuantParam);
#else
            MLSL_FREE(globalQuantParam);
#endif
        globalQuantParam = NULL;

        isInitialized = false;
        MLSL_LOG(DEBUG, "finalized");
    }

    bool Environment::IsInitialized() { return isInitialized; }
    size_t Environment::GetProcessIdx() { return globalProcessGroup->GetIdx(); }
    size_t Environment::GetProcessCount() { return globalProcessGroup->GetSize(); }
    int Environment::GetVersion() { return MLSL_VERSION(MLSL_MAJOR_VERSION, MLSL_MINOR_VERSION); }
    void* Environment::Alloc(size_t sz, size_t alignment) { return CommAlloc(sz, alignment); }
    void Environment::Free(void* p) { CommFree(p); }

    Session* Environment::CreateSession(PhaseType phaseType)
    {
        SessionImpl* si = new SessionImpl(phaseType);
        return si;
    }

    void Environment::DeleteSession(Session* session)
    {
        SessionImpl* p = static_cast<SessionImpl*>(session);
        delete p;
    }

    Distribution* Environment::CreateDistribution(size_t dataPartitions, size_t modelPartitions)
    {
        DistributionImpl* di = new DistributionImpl(dataPartitions, modelPartitions, true /*replicate*/, -1, -1);
        return di;
    }

    Distribution* Environment::CreateDistributionWithColors(int dataColor, int modelColor)
    {
        DistributionImpl* di = new DistributionImpl(-1, -1, false /*replicate*/, dataColor, modelColor);
        return di;
    }

    void Environment::DeleteDistribution(Distribution* dist)
    {
        DistributionImpl* p = static_cast<DistributionImpl*>(dist);
        delete p;
    }

    void Environment::Wait(CommReq* commReq)
    {
        CommRequest* req = (CommRequest*)commReq;
        req->Wait(false);
        RequestStorage::GetObject().RemoveRequest(req);
    }

    void Environment::Test(CommReq* commReq, bool* isCompleted)
    {
        CommRequest* req = (CommRequest*)commReq;
        req->Test(isCompleted, false);
        if (*isCompleted == true) RequestStorage::GetObject().RemoveRequest(req);
    }

    void Environment::SetQuantizationParams(QuantParams* qparams)
    {
        MLSL_ASSERT(!globalQuantParam, "quantization parameters can be set only once");
#ifdef USE_EPLIB
        globalQuantParam = (QuantParams*)EPLIB_quant_params_submit((void*)qparams);
#else
        if (globalQuantParam) MLSL_FREE(globalQuantParam);
        globalQuantParam = (QuantParams*)MLSL_MALLOC(sizeof(QuantParams), CACHELINE_SIZE);
#endif
    }

    QuantParams* Environment::GetQuantizationParams()
    {
        return globalQuantParam;
    }
};

