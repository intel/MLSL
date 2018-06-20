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
#ifndef COMM_HPP
#define COMM_HPP

/* Internal Comm Interface for MLSL */

#include <string>
#include <vector>

#include "log.hpp"
#include "mlsl.hpp"

namespace MLSL
{
    using namespace std;
    class ProcessGroupImpl;
    class CommRequestImpl;

    class ProcessGroup
    {
    private:
        ProcessGroupImpl* p;
        size_t idx;
        size_t size;
    public:
        ProcessGroup(ProcessGroupImpl* p_, size_t idx_, size_t size_) : p(p_), idx(idx_), size(size_) { }
        ~ProcessGroup() { p = NULL; }

        ProcessGroupImpl* GetImpl() { return p; }
        size_t GetIdx() { return idx; }
        size_t GetSize() { return size; }
    };

    class CommOp
    {
    public:

        enum ReqType
        {
            None          = -1,
            AllGather     = 0,
            ReduceScatter = 1,
            AllReduce     = 2,
            AlltoAll      = 3,
            Barrier       = 4,
            Bcast         = 5,
            Reduce        = 6,
            Gather        = 7,
            Scatter       = 8,
            AlltoAllv     = 9,
            AllGatherv    = 10,
            SendRecvList  = 100
        };

        enum ReduceOp
        {
            RO_SUM = 0,
            RO_MIN = 1,
            RO_MAX = 2
        };

    protected:
        ReqType reqType;            // allgather, reduce_scatter, allreduce, reshape
        ProcessGroup* processGroup; // ProcessGroup to work on
        size_t len;                 // length in datatype's elements
    public:
        CommOp(ReqType reqType_, ProcessGroup* pGroup, size_t length)
               : reqType(reqType_), processGroup(pGroup), len(length) { }
        virtual ~CommOp() {}

        ReqType GetReqType() { return reqType; }
        ProcessGroup* GetProcessGroup() { return processGroup; }
        size_t GetLen() { return len; }

        string GetReqName()
        {
            string reqName;
            if (reqType == AllGather)          reqName = "AllGather";
            else if (reqType == ReduceScatter) reqName = "ReduceScatter";
            else if (reqType == AllReduce)     reqName = "AllReduce";
            else if (reqType == AlltoAll)      reqName = "AlltoAll";
            else if (reqType == SendRecvList)  reqName = "SendRecvList";
            else if (reqType == Barrier)       reqName = "Barrier";
            else if (reqType == Bcast)         reqName = "Broadcast";
            else if (reqType == Reduce)        reqName = "Reduce";
            else if (reqType == Gather)        reqName = "Gather";
            else if (reqType == Scatter)       reqName = "Scatter";
            else if (reqType == AlltoAllv)     reqName = "AlltoAllv";
            else if (reqType == AllGatherv)    reqName = "AllGatherv";
            return reqName;
        }
    };

    class CommOpAllGather : public CommOp
    {
    public:
        CommOpAllGather(size_t length, ProcessGroup* pGroup) : CommOp(AllGather, pGroup, length) { }
    };

    class CommOpAllGatherv : public CommOp
    {
    protected:
        size_t* recvCounts;
    public:
        CommOpAllGatherv(size_t length, ProcessGroup* pGroup, size_t* rcounts) : CommOp(AllGatherv, pGroup, length), recvCounts(rcounts) { }
        size_t* GetRecvCounts() { return recvCounts; }
    };

    class CommOpReduceScatter : public CommOp
    {
    protected:
        ReduceOp redOp;
    public:
        CommOpReduceScatter(size_t length, ProcessGroup* pGroup, ReduceOp redOp_ = RO_SUM)
                             : CommOp(ReduceScatter, pGroup, length), redOp(redOp_) { }
        ReduceOp GetOp() { return redOp; }
    };

    class CommOpAllReduce : public CommOp
    {
    protected:
        ReduceOp redOp;
    public:
        CommOpAllReduce(size_t length, ProcessGroup* pGroup, ReduceOp redOp_ = RO_SUM)
                         : CommOp(AllReduce, pGroup, length), redOp(redOp_) { }
        ReduceOp GetOp() { return redOp; }
    };

    class CommOpReduce : public CommOp
    {
    protected:
        ReduceOp redOp;
        size_t rootIdx;
    public:
        CommOpReduce(size_t length, ProcessGroup* pGroup, size_t _rootIdx, ReduceOp redOp_ = RO_SUM)
                         : CommOp(Reduce, pGroup, length), redOp(redOp_), rootIdx(_rootIdx) { }
        ReduceOp GetOp() { return redOp; }
        size_t GetRootIdx() { return rootIdx; }
    };

    class CommOpGather : public CommOp
    {
    protected:
        size_t rootIdx;
    public:
        CommOpGather(size_t length, ProcessGroup* pGroup, size_t _rootIdx)
                         : CommOp(Gather, pGroup, length), rootIdx(_rootIdx) { }
        size_t GetRootIdx() { return rootIdx; }
    };

    class CommOpScatter : public CommOp
    {
    protected:
        size_t rootIdx;
    public:
        CommOpScatter(size_t length, ProcessGroup* pGroup, size_t _rootIdx)
                         : CommOp(Scatter, pGroup, length), rootIdx(_rootIdx) { }
        size_t GetRootIdx() { return rootIdx; }
    };

    class CommOpAlltoAll : public CommOp
    {
    public:
        CommOpAlltoAll(size_t length, ProcessGroup* pGroup) : CommOp(AlltoAll, pGroup, length) { }
    };

    class CommOpAlltoAllv : public CommOp
    {
    protected:
        size_t* sendCounts;
        size_t* recvCounts;
        size_t* sendOffsets;
        size_t* recvOffsets;
    public:
        CommOpAlltoAllv(size_t length, ProcessGroup* pGroup, size_t* scounts, size_t* soffsets, size_t* rcounts, size_t* roffsets)
                       : CommOp(AlltoAllv, pGroup, length), sendCounts(scounts), recvCounts(rcounts), sendOffsets(soffsets), recvOffsets(roffsets) { }

        size_t* GetSendCounts() { return sendCounts; }
        size_t* GetRecvCounts() { return recvCounts; }
        size_t* GetSendOffsets() { return sendOffsets; }
        size_t* GetRecvOffsets() { return recvOffsets; }
    };

    class CommOpBarrier : public CommOp
    {
    public:
        CommOpBarrier(ProcessGroup* pGroup) : CommOp(Barrier, pGroup, 0) { }
    };

    class CommOpBcast : public CommOp
    {
        size_t rootIdx;
    public:
        size_t GetRootIdx() { return rootIdx; }
        CommOpBcast(size_t length, ProcessGroup* pGroup, size_t _rootIdx) : CommOp(Bcast, pGroup, length), rootIdx(_rootIdx) { }
    };

    class CommOpSRList : public CommOp
    {
    public:
        class SRInfo
        {
        private:
            size_t length;
            size_t peer;
            size_t offset;
        public:
            SRInfo(size_t l, size_t p, size_t o) : length(l), peer(p), offset(o) { }
            size_t GetLen() { return length; }
            size_t GetPeer() { return peer; }
            size_t GetOffset() { return offset; }
        };
    protected:
        vector<SRInfo> sendList;
        vector<SRInfo> recvList;
    public:
        CommOpSRList(ProcessGroup* pGroup) : CommOp(SendRecvList, pGroup, 0) { }
        int AddSend(size_t len, size_t dst, size_t offset)
        {
            SRInfo si(len, dst, offset);
            sendList.push_back(si);
            return 0;
        }
        int AddRecv(size_t len, size_t src, size_t offset)
        {
            SRInfo ri(len, src, offset);
            recvList.push_back(ri);
            return 0;
        }
        size_t GetSendCount() { return sendList.size(); }
        size_t GetRecvCount() { return recvList.size(); }
        const SRInfo& GetSendInfo(size_t idx) { return sendList.at(idx); }
        const SRInfo& GetRecvInfo(size_t idx) { return recvList.at(idx); }
    };

    class CommDesc
    {
    public:
        enum CompType
        {
            NONE       = -1,
            FPROP      = 0,
            BPROP      = 1,
            PARAM_GRAD = 2,
            PARAM_INC  = 3,
            GENERIC    = 4
        };

    protected:
        DataType dataType;    // DT_FLOAT, DT_DOUBLE
        int opUniqueId;       // ComputeOp's unique id
        CompType compType;    // fprop, bprop, delwt, wtinc
        vector<CommOp*> ops;

        void AddOp(CommOp* op) { ops.push_back(op); }
    public:
        CommDesc(DataType dataType_, int opUniqueId_, CompType compType_)
                  : dataType(dataType_), opUniqueId(opUniqueId_), compType(compType_) { }
        ~CommDesc()
        {
            for (size_t idx = 0; idx < ops.size(); idx++)
                delete ops[idx];
            ops.clear();
        }
        string GetCompName()
        {
            string n = "Unknown";
            if (compType == FPROP)           n = "FPROP";
            else if (compType == BPROP)      n = "BPROP";
            else if (compType == PARAM_GRAD) n = "PARAM_GRAD";
            else if (compType == PARAM_INC)  n = "PARAM_INC";
            return n;
        }

        CommOpAllGather* AddAllGather(size_t length, ProcessGroup* pGroup)
        {
            CommOpAllGather* op = new CommOpAllGather(length, pGroup);
            ops.push_back(op);
            return op;
        }
        CommOpAllGatherv* AddAllGatherv(size_t length, ProcessGroup* pGroup, size_t* recvCounts)
        {
            CommOpAllGatherv* op = new CommOpAllGatherv(length, pGroup, recvCounts);
            ops.push_back(op);
            return op;
        }
        CommOpReduceScatter* AddReduceScatter(size_t length, ProcessGroup* pGroup, CommOp::ReduceOp redOp = CommOp::RO_SUM)
        {
            CommOpReduceScatter* op = new CommOpReduceScatter(length, pGroup, redOp);
            ops.push_back(op);
            return op;
        }
        CommOpAllReduce* AddAllReduce(size_t length, ProcessGroup* pGroup, CommOp::ReduceOp redOp = CommOp::RO_SUM)
        {
            CommOpAllReduce* op = new CommOpAllReduce(length, pGroup, redOp);
            ops.push_back(op);
            return op;
        }
        CommOpReduce* AddReduce(size_t length, ProcessGroup* pGroup, size_t rootIdx, CommOp::ReduceOp redOp = CommOp::RO_SUM)
        {
            CommOpReduce* op = new CommOpReduce(length, pGroup, rootIdx, redOp);
            ops.push_back(op);
            return op;
        }
        CommOpGather* AddGather(size_t length, ProcessGroup* pGroup, size_t rootIdx)
        {
            CommOpGather* op = new CommOpGather(length, pGroup, rootIdx);
            ops.push_back(op);
            return op;
        }
        CommOpScatter* AddScatter(size_t length, ProcessGroup* pGroup, size_t rootIdx)
        {
            CommOpScatter* op = new CommOpScatter(length, pGroup, rootIdx);
            ops.push_back(op);
            return op;
        }
        CommOpAlltoAll* AddAlltoAll(size_t length, ProcessGroup* pGroup)
        {
            CommOpAlltoAll* op = new CommOpAlltoAll(length, pGroup);
            ops.push_back(op);
            return op;
        }
        CommOpAlltoAllv* AddAlltoAllv(size_t length, ProcessGroup* pGroup, size_t* scounts, size_t* soffsets, size_t* rcounts, size_t* roffsets)
        {
            CommOpAlltoAllv* op = new CommOpAlltoAllv(length, pGroup, scounts, soffsets, rcounts, roffsets);
            ops.push_back(op);
            return op;
        }
        CommOpBcast* AddBcast(size_t length, ProcessGroup* pGroup, size_t rootIdx)
        {
            CommOpBcast* op = new CommOpBcast(length, pGroup, rootIdx);
            ops.push_back(op);
            return op;
        }
        CommOpBarrier* AddBarrier(ProcessGroup* pGroup)
        {
            CommOpBarrier* op = new CommOpBarrier(pGroup);
            ops.push_back(op);
            return op;
        }
        CommOpSRList* AddSRList(ProcessGroup* pGroup)
        {
            CommOpSRList* op = new CommOpSRList(pGroup);
            ops.push_back(op);
            return op;
        }
        DataType GetDataType() { return dataType; }
        CompType GetComputeType() { return compType; }
        int GetOpUniqueId() { return opUniqueId; }
        size_t GetOpCount() { return ops.size(); }
        CommOp* GetOp(size_t idx) { return ops.at(idx); }
    };

    class CommRequest
    {
        friend class CommRequestImpl;
    private:
        CommRequestImpl* p;
        bool isStarted;
        bool isCompleted;
        void* bufPtr;
        size_t tmpSz;
        bool isInPlace;
        CommDesc* desc;

        CommRequest(const CommRequest& cr);
        CommRequest& operator=(const CommRequest& cr);

    public:
        CommRequest(CommRequestImpl* p_, DataType dataType, int opUniqueId, CommDesc::CompType compType)
                     : p(p_), isStarted(false), isCompleted(false), bufPtr(NULL), tmpSz(0), isInPlace(true)
        {
            desc = new CommDesc(dataType, opUniqueId, compType);
        }
        ~CommRequest()
        {
            if (desc)
            {
                delete desc;
                desc = NULL;
            }
        }
        CommDesc* GetDesc() { return desc; }
        void Setup();
        size_t GetBufSize() { return tmpSz; }
        void* GetBuf();
        bool IsStarted();
        bool IsInPlace() { return isInPlace; }
        void SetInPlace(bool inplace) { isInPlace = inplace; }
        int Start(void* buf, void* tmpBuf);
        void* Wait(bool resetPtr = true);
        void* Test(bool* isCompleted, bool resetPtr = false);
        CommRequestImpl* GetImpl() { return p; }
        void SetCompressionType(CompressionType compressType);
    };

    ProcessGroup* GetGlobalProcessGroup();
    ProcessGroup* GetSelfProcessGroup();
    ProcessGroup* CreateProcessGroup(int color);
    ProcessGroup* SplitProcessGroup(ProcessGroup* pg, int color);
    void SetGlobalProcessGroup(ProcessGroup* pg);
    void FreeProcessGroup(ProcessGroup* pGroup);

    int CommInit(int* argc, char** argv[]);
    int CommFinalize();
    void CommBarrier();
    void* CommAlloc(size_t sz, size_t alignment);
    void CommFree(void* ptr);
    CommRequest* CommCreateRequest(DataType dataType, int opId, CommDesc::CompType compType);
    void CommFreeRequest(CommRequest* req);
};

#endif /* COMM_HPP */
