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
#include <algorithm>
#include <numeric>   /* std::accumulate */

#include "common.hpp"
#include "mlsl_impl.hpp"

#define BUFFER_SIZE    65536
#define KBYTE          1024
#define KILO           1000.0

#define BATCH_FORMAT         " mini-batch count (number of iterations): %zu\n global mini-batch size (number of images in mini-batch): %zu\n total number of images: %zu\n"
#define FORMAT_HEADER_ROW    "format: [total comm size (Kbytes), comm cycles per image (Kcycles/img)] for each IA/OA/GRAD/INC\n"
#define HEADER_DELIMETER     "-----------------"
#define OP_NAME              "op name"
#define ENTITY_HEADER_FORMAT "%4s #%-3zu             | "
#define ENTITY_VALUE_FORMAT  "%10zu %10.1f | "

#define PRINTF_PARAMS outFile, tbuf + c, m - c

namespace MLSL
{
    StatisticsImpl::StatisticsImpl(Session* s)
                      : session(static_cast<SessionImpl*>(s)),
                        totalCompTime(0),
                        totalCommTime(0),
                        totalCommSize(0),
                        totalIsolationCommTime(0),
                        totalIsolationCommSize(0),
                        globalTime(0),
                        iterations(10),
                        skip(4),
                        isStarted(false),
                        batchCount(0),
                        startOpMonitor(-1),
                        maxIACount(0),
                        maxOACount(0),
                        maxParamCount(0),
                        isIACommSet(false),
                        isOACommSet(false),
                        isParamCommSet(false),
                        isStatsEnabled(false)
    {
        MLSL_ASSERT(session, "session is null");
        isStatsEnabled = envData.enableStats;
    }

    void StatisticsImpl::Initialize()
    {
        if (!isStatsEnabled) return;

        stats.clear();
        stats.resize(session->GetOperationCount());

        for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
        {
            Operation* op = session->GetOperation(opIdx);
            size_t entityCount = op->GetInputCount() + op->GetOutputCount() + (op->GetParameterSetCount() * 2);

            stats[opIdx].isolationCommStartTimes.resize(entityCount);
            stats[opIdx].isolationCommWaitTimes.resize(entityCount);
            stats[opIdx].compStartTimes.resize(entityCount);
            stats[opIdx].compWaitTimes.resize(entityCount);
            stats[opIdx].commStartTimes.resize(entityCount);
            stats[opIdx].commWaitTimes.resize(entityCount);

            stats[opIdx].isolationCommSizes.resize(entityCount);
            stats[opIdx].isolationCommSizesPerBatch.resize(entityCount);
            stats[opIdx].commSizes.resize(entityCount);
            stats[opIdx].isTested.resize(entityCount);

            stats[opIdx].isolationCommTime = stats[opIdx].isolationCommSize =
            stats[opIdx].commTime = stats[opIdx].commSize = stats[opIdx].compTime = 0;

        }
        totalIsolationCommTime = totalIsolationCommSize = globalTime = 0;
        Reset();
    }

    void StatisticsImpl::PrintIsolationComm(FILE* outFile)
    {
        if (!isStatsEnabled) return;

        int c = 0, m = BUFFER_SIZE;
        char tbuf[BUFFER_SIZE] = { 0 };

        /* for now lets print only from process id 0 */
        if (globalProcessGroup->GetIdx()) return;

        /* lets print the run-time comp and comm */
        size_t isoBatchCount = iterations - skip;

        /* batchSize over all isolation iterations */
        size_t isoBatchSize = session->GetGlobalMinibatchSize() * isoBatchCount;
        if (isoBatchCount >= 1)
        {
            c += mysnprintf(PRINTF_PARAMS, "\nstatistics in isolation environment (computation OFF)\n");
            c += mysnprintf(PRINTF_PARAMS, BATCH_FORMAT, isoBatchCount, session->GetGlobalMinibatchSize(), isoBatchSize);
            c += mysnprintf(PRINTF_PARAMS, FORMAT_HEADER_ROW);

            size_t entityCount = 0;
            if (isIACommSet)
            {
                for (size_t actIdx = 0; actIdx < maxIACount; actIdx++)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "IA", actIdx);
                    entityCount++;
                }
            }

            if (isOACommSet)
            {
                for (size_t actIdx = 0; actIdx < maxOACount; actIdx++)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "OA", actIdx);
                    entityCount++;
                }
            }

            for (size_t paramIdx = 0; paramIdx < maxParamCount; paramIdx++)
            {
                c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "GRAD", paramIdx);
                entityCount++;
                if (isParamCommSet)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "INC", paramIdx);
                    entityCount++;
                }
            }
            c += mysnprintf(PRINTF_PARAMS, OP_NAME"\n");
            entityCount++;

            for (size_t entIdx = 0; entIdx < entityCount; entIdx++)
                c += mysnprintf(PRINTF_PARAMS, HEADER_DELIMETER);
            c += mysnprintf(PRINTF_PARAMS, "\n");

            for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
            {
                Operation* op = session->GetOperation(opIdx);
                size_t statIdx = 0;
                for (size_t actIdx = 0; actIdx < maxIACount; actIdx++)
                {
                    if (actIdx >= op->GetInputCount())
                    {
                        if (isIACommSet) c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        if (isIACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].isolationCommSizes[statIdx] / KBYTE),
                                (stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx]) / (isoBatchSize * KILO));
                        statIdx++;
                    }
                }
                for (size_t actIdx = 0; actIdx < maxOACount; actIdx++)
                {
                    if (actIdx >= op->GetOutputCount())
                    {
                        if (isOACommSet) c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        if (isOACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].isolationCommSizes[statIdx] / KBYTE),
                                (stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx]) / (isoBatchSize * KILO));
                        statIdx++;
                    }
                }
                for (size_t paramIdx = 0; paramIdx < maxParamCount; paramIdx++)
                {
                    if (paramIdx >= op->GetParameterSetCount())
                    {
                        c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                        if (isParamCommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                            (stats[opIdx].isolationCommSizes[statIdx] / KBYTE),
                            (stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx]) / (isoBatchSize * KILO));
                        statIdx++;
                        if (isParamCommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].isolationCommSizes[statIdx] / KBYTE),
                                (stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx]) / (isoBatchSize * KILO));
                        statIdx++;
                    }
                }
                c += mysnprintf(PRINTF_PARAMS, "%8s\n", session->GetOperation(opIdx)->GetName());
            }

            double totalIsolationCommTimePerImg = (double)totalIsolationCommTime / isoBatchSize;

            c += mysnprintf(PRINTF_PARAMS, "comm cycles per image = %.1f сycles/img (%.1f Kсycles/img)\n",
                            totalIsolationCommTimePerImg, totalIsolationCommTimePerImg / KILO);
            c += mysnprintf(PRINTF_PARAMS, "total comm cycles = %llu cycles (%.1f Kcycles)\n",
                            totalIsolationCommTime, (double)totalIsolationCommTime / KILO);
            c += mysnprintf(PRINTF_PARAMS, "total comm size = %zu bytes (%zu Kbytes)\n",
                            totalIsolationCommSize, totalIsolationCommSize / KBYTE);
        }

        tbuf[BUFFER_SIZE - 1] = 0;
        printf("%s", tbuf);
    }

    void StatisticsImpl::Print(FILE* outFile)
    {
        if (!isStatsEnabled) return;

        int c = 0, m = BUFFER_SIZE;
        char tbuf[BUFFER_SIZE] = { 0 };

        /* for now lets print only from process id 0 */
        if (globalProcessGroup->GetIdx()) return;

        /* lets print the run-time comp and comm */

        /* batchSize over all iterations */
        size_t batchSize = session->GetGlobalMinibatchSize() * batchCount;
        if (batchCount >= 1)
        {
            c += mysnprintf(PRINTF_PARAMS, "\nstatistics in real environment (computation ON)\n");
            c += mysnprintf(PRINTF_PARAMS, BATCH_FORMAT,
                            batchCount, session->GetGlobalMinibatchSize(), batchSize);
            c += mysnprintf(PRINTF_PARAMS, FORMAT_HEADER_ROW);

            size_t entityCount = 0;
            if (isIACommSet)
                for (size_t actIdx = 0; actIdx < maxIACount; actIdx++)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "IA", actIdx);
                    entityCount++;
                }

            if (isOACommSet)
                for (size_t actIdx = 0; actIdx < maxOACount; actIdx++)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "OA", actIdx);
                    entityCount++;
                }

            for (size_t paramIdx = 0; paramIdx < maxParamCount; paramIdx++)
            {
                c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "GRAD", paramIdx);
                entityCount++;
                if (isParamCommSet)
                {
                    c += mysnprintf(PRINTF_PARAMS, ENTITY_HEADER_FORMAT, "INC", paramIdx);
                    entityCount++;
                }
            }

            c += mysnprintf(PRINTF_PARAMS, OP_NAME"\n");
            entityCount++;

            for (size_t entIdx = 0; entIdx < entityCount; entIdx++)
                c += mysnprintf(PRINTF_PARAMS, HEADER_DELIMETER);
            c += mysnprintf(PRINTF_PARAMS, "\n");

            for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
            {
                Operation* op = session->GetOperation(opIdx);
                size_t statIdx = 0;
                for (size_t actIdx = 0; actIdx < maxIACount; actIdx++)
                {
                    if (actIdx >= op->GetInputCount())
                    {
                        if (isIACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        if (isIACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].commSizes[statIdx] / KBYTE),
                                (stats[opIdx].commStartTimes[statIdx] + stats[opIdx].commWaitTimes[statIdx]) / (batchSize * KILO));
                        statIdx++;
                    }
                }

                for (size_t actIdx = 0; actIdx < maxOACount; actIdx++)
                {
                    if (actIdx >= op->GetOutputCount())
                    {
                        if (isOACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        if (isOACommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].commSizes[statIdx] / KBYTE),
                                (stats[opIdx].commStartTimes[statIdx] + stats[opIdx].commWaitTimes[statIdx]) / (batchSize * KILO));
                        statIdx++;
                    }
                }

                for (size_t paramIdx = 0; paramIdx < maxParamCount; paramIdx++)
                {
                    if (paramIdx >= op->GetParameterSetCount())
                    {
                        c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                        if (isParamCommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT, 0, 0);
                    }
                    else
                    {
                        c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                            (stats[opIdx].commSizes[statIdx] / KBYTE),
                            (stats[opIdx].commStartTimes[statIdx] + stats[opIdx].commWaitTimes[statIdx]) / (batchSize * KILO));
                        statIdx++;
                        if (isParamCommSet)
                            c += mysnprintf(PRINTF_PARAMS, ENTITY_VALUE_FORMAT,
                                (stats[opIdx].commSizes[statIdx] / KBYTE),
                                (stats[opIdx].commStartTimes[statIdx] + stats[opIdx].commWaitTimes[statIdx]) / (batchSize * KILO));
                        statIdx++;
                    }
                }
                c += mysnprintf(PRINTF_PARAMS, "%8s\n", session->GetOperation(opIdx)->GetName());
            }

            double totalCommTimePerImg = (double)totalCommTime / batchSize;
            double totalCompTimePerImg = (double)totalCompTime / batchSize;
            double totalTimePerImg = (totalCommTimePerImg + totalCompTimePerImg);

            c += mysnprintf(PRINTF_PARAMS, "comm cycles per image = %.1f cycles/img (%.1f Kсycles/img)\n",
                           totalCommTimePerImg, totalCommTimePerImg / KILO);
            c += mysnprintf(PRINTF_PARAMS, "comp cycles per image = %.1f cycles/img (%.1f Kсycles/img)\n",
                           totalCompTimePerImg, totalCompTimePerImg / KILO);
            c += mysnprintf(PRINTF_PARAMS, "(comm + comp) cycles per image = %.1f cycles/img (%.1f Kсycles/img)\n",
                            totalTimePerImg, totalTimePerImg / KILO);

            c += mysnprintf(PRINTF_PARAMS, "total comm cycles = %llu cycles (%.1f Kcycles)\n",
                            totalCommTime, (double)totalCommTime / KILO);
            c += mysnprintf(PRINTF_PARAMS, "total compute cycles = %llu cycles (%.1f Kcycles)\n",
                            totalCompTime, (double)totalCompTime / KILO);
            c += mysnprintf(PRINTF_PARAMS, "total comm size = %zu bytes (%zu Kbytes)\n",
                            totalCommSize, totalCommSize / KBYTE);
        }
        tbuf[BUFFER_SIZE - 1] = 0;
        printf("%s", tbuf);
        PrintIsolationComm(outFile);
    }

    void StatisticsImpl::Reset()
    {
        if (!isStatsEnabled) return;

        MLSL_ASSERT(isStarted == false, "stop collection before resetting");

        for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
        {
            std::fill(stats[opIdx].commStartTimes.begin(), stats[opIdx].commStartTimes.end(), 0);
            std::fill(stats[opIdx].commWaitTimes.begin(), stats[opIdx].commWaitTimes.end(), 0);
            std::fill(stats[opIdx].compStartTimes.begin(), stats[opIdx].compStartTimes.end(), 0);
            std::fill(stats[opIdx].compWaitTimes.begin(), stats[opIdx].compWaitTimes.end(), 0);
            std::fill(stats[opIdx].commSizes.begin(), stats[opIdx].commSizes.end(), 0);
            std::fill(stats[opIdx].isTested.begin(), stats[opIdx].isTested.end(), false);
            stats[opIdx].commSize = stats[opIdx].compTime = stats[opIdx].commTime = 0;
        }
        totalCommSize = 0;
        totalCompTime = 0;
        totalCommTime = 0;
        batchCount = 0;
    }

    void StatisticsImpl::CollectIsolationStats()
    {
        if (!isStatsEnabled) return;

        unsigned long long startTime = 0, endTime = 0;
        float* tmpBuf;
        size_t maxSize = 1;

        for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
        {
            std::fill(stats[opIdx].isolationCommStartTimes.begin(), stats[opIdx].isolationCommStartTimes.end(), 0);
            std::fill(stats[opIdx].isolationCommWaitTimes.begin(), stats[opIdx].isolationCommWaitTimes.end(), 0);
            std::fill(stats[opIdx].isolationCommSizes.begin(), stats[opIdx].isolationCommSizes.end(), 0);
            std::fill(stats[opIdx].isolationCommSizesPerBatch.begin(), stats[opIdx].isolationCommSizesPerBatch.end(), 0);
            stats[opIdx].isolationCommTime = stats[opIdx].isolationCommSize = 0;
        }
        totalIsolationCommTime = 0;
        totalIsolationCommSize = 0;

        Reset();

        for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
        {
            Operation* op = session->GetOperation(opIdx);
            if (maxOACount < op->GetOutputCount()) maxOACount = op->GetOutputCount();
            if (maxIACount < op->GetInputCount()) maxIACount = op->GetInputCount();
            if (maxParamCount < op->GetParameterSetCount()) maxParamCount = op->GetParameterSetCount();

            for (size_t paramIdx = 0; paramIdx < op->GetParameterSetCount(); paramIdx++)
            {
                ParameterSetImpl* pParameterSetImpl = static_cast<ParameterSetImpl*>(op->GetParameterSet(paramIdx));
                if (maxSize < pParameterSetImpl->GetCommBuf()->GetSize()) maxSize = pParameterSetImpl->GetCommBuf()->GetSize();
                if (startOpMonitor < 0) startOpMonitor = opIdx;
            }
        }

        tmpBuf = (float*)Environment::GetEnv().Alloc(maxSize * sizeof(float), 4096);
        MLSL_ASSERT(tmpBuf, "cannot allocate %zu", maxSize);

        for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
        {
            Operation* op = session->GetOperation(opIdx);
            size_t statIdx = op->GetInputCount();

            for (size_t actIdx = 0; actIdx < op->GetOutputCount(); actIdx++, statIdx++)
            {
                ActivationImpl* actImpl = static_cast<ActivationImpl*>(op->GetOutput(actIdx));
                if (!actImpl->IsCommRequired()) continue;
                Activation* act = op->GetOutput(actIdx);
                isOACommSet = true;
                ActivationImpl* peerAct = actImpl->GetPeer();
                float* commInputBuf = (float*)act->GetCommBuf();
                if (peerAct)
                {
                    stats[opIdx].isolationCommSizesPerBatch[statIdx] = actImpl->GetMsgSize();
                    for (size_t iter = 0; iter < iterations; iter++)
                    {
                        if (iter == skip)
                        {
                            stats[opIdx].isolationCommStartTimes[statIdx] = 0;
                            stats[opIdx].isolationCommWaitTimes[statIdx] = 0;
                        }
                        startTime = rdtsc();
                        act->StartComm(commInputBuf);
                        endTime = rdtsc();
                        stats[opIdx].isolationCommStartTimes[statIdx] += (endTime - startTime);

                        startTime = rdtsc();
                        peerAct->WaitComm();
                        endTime = rdtsc();
                        stats[opIdx].isolationCommWaitTimes[statIdx] += (endTime - startTime);
                    }
                    stats[opIdx].isolationCommTime += (stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx]);
                }
            }

            statIdx = 0;
            for (size_t actIdx = 0; actIdx < op->GetInputCount(); actIdx++, statIdx++)
            {
                ActivationImpl* actImpl = static_cast<ActivationImpl*>(op->GetInput(actIdx));
                if (!actImpl->IsCommRequired()) continue;
                Activation* act = op->GetInput(actIdx);
                isIACommSet = true;
                ActivationImpl* peerAct = actImpl->GetPeer();
                float* commInputBuf = (float*)act->GetCommBuf();
                if (peerAct)
                {
                    stats[opIdx].isolationCommSizesPerBatch[statIdx] = actImpl->GetMsgSize();
                    for (size_t iter = 0; iter < iterations; iter++)
                    {
                        if (iter == skip)
                        {
                            stats[opIdx].isolationCommStartTimes[statIdx] = 0;
                            stats[opIdx].isolationCommWaitTimes[statIdx] = 0;
                        }
                        startTime = rdtsc();
                        act->StartComm(commInputBuf);
                        endTime = rdtsc();
                        stats[opIdx].isolationCommStartTimes[statIdx] += (endTime - startTime);

                        startTime = rdtsc();
                        peerAct->WaitComm();
                        endTime = rdtsc();
                        stats[opIdx].isolationCommWaitTimes[statIdx] += (endTime - startTime);
                    }
                    stats[opIdx].isolationCommTime += stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx];
                }
            }

            statIdx = op->GetInputCount() + op->GetOutputCount();
            for (size_t paramIdx = 0; paramIdx < op->GetParameterSetCount(); paramIdx++)
            {
                OperationImpl* opImpl = static_cast<OperationImpl*>(op);
                ParameterSetImpl* paramImpl = opImpl->GetParameterSet(paramIdx);
                ParameterSet* param = op->GetParameterSet(paramIdx);
                stats[opIdx].isolationCommSizesPerBatch[statIdx] = paramImpl->GetGradientMsgSize();
                if (paramImpl->IsCommRequired())
                    for (size_t iter = 0; iter < iterations; iter++)
                    {
                        if (iter == skip)
                        {
                            stats[opIdx].isolationCommStartTimes[statIdx] = 0;
                            stats[opIdx].isolationCommWaitTimes[statIdx] = 0;
                        }

                        startTime = rdtsc();
                        param->StartGradientComm(tmpBuf);
                        endTime = rdtsc();
                        stats[opIdx].isolationCommStartTimes[statIdx] += (endTime - startTime);

                        startTime = rdtsc();
                        param->WaitGradientComm();
                        endTime = rdtsc();
                        stats[opIdx].isolationCommWaitTimes[statIdx] += (endTime - startTime);
                    }
                stats[opIdx].isolationCommTime += stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx];
                statIdx++;
                stats[opIdx].isolationCommSizesPerBatch[statIdx] = paramImpl->GetIncrementMsgSize();
                if (paramImpl->IsDistributedUpdate())
                {
                    isParamCommSet = true;
                    for (size_t iter = 0; iter < iterations; iter++)
                    {
                        if (iter == skip)
                        {
                            stats[opIdx].isolationCommStartTimes[statIdx] = 0;
                            stats[opIdx].isolationCommWaitTimes[statIdx] = 0;
                        }

                        startTime = rdtsc();
                        param->StartIncrementComm(tmpBuf);
                        endTime = rdtsc();
                        stats[opIdx].isolationCommStartTimes[statIdx] += (endTime - startTime);

                        startTime = rdtsc();
                        param->WaitIncrementComm();
                        endTime = rdtsc();
                        stats[opIdx].isolationCommWaitTimes[statIdx] += (endTime - startTime);
                    }
                    stats[opIdx].isolationCommTime += stats[opIdx].isolationCommStartTimes[statIdx] + stats[opIdx].isolationCommWaitTimes[statIdx];
                }
                statIdx++;
            }

            for (size_t idx = 0; idx < stats[opIdx].isolationCommSizesPerBatch.size(); idx++)
                stats[opIdx].isolationCommSizes[idx] = stats[opIdx].isolationCommSizesPerBatch[idx] * (iterations - skip);

            stats[opIdx].isolationCommSize = std::accumulate(stats[opIdx].isolationCommSizes.begin(),
                                                             stats[opIdx].isolationCommSizes.end(),
                                                             size_t(0));

            totalIsolationCommTime += stats[opIdx].isolationCommTime;
            totalIsolationCommSize += stats[opIdx].isolationCommSize;
        }
        Environment::GetEnv().Free(tmpBuf);
    }

    void StatisticsImpl::UpdateStats(const StatEvent& e)
    {
        if (!isStatsEnabled) return;

        OperationImpl* op = static_cast<OperationImpl*>(session->GetOperation(e.opIdx));
        StatisticsImpl* statsImpl = static_cast<StatisticsImpl*>(op->GetSession()->GetStats());
        if (statsImpl->IsStarted())
        {
            size_t statIdx = e.entIdx;
            size_t opIdx = op->GetOpIndex();

            if (!e.isParam)
            {
                /* 'start' and 'wait' timings should be accounted for the same activation (which started communication) */
                if (!e.isInputOrIncrement)
                {
                    if (e.actionType == StatEvent::Start)
                    {
                        /* start communication for output activation, forward pass */
                        statIdx = op->GetInputCount() + e.entIdx;
                    }
                    else
                    {
                        /* wait communication for output activation, backward pass */
                        ActivationImpl* act = op->GetOutput(e.entIdx);
                        ActivationImpl* peerAct = act->GetPeer();
                        if (!peerAct) return;
                        OperationImpl* peerOp = peerAct->GetOp();
                        opIdx = peerOp->GetOpIndex();
                        statIdx = peerAct->GetActIndex();
                    }
                }
                else
                {
                    if (e.actionType == StatEvent::Start)
                    {
                        /* start communication for input activation, backward pass */
                        /* no actions, we already have correct statIdx and opIdx */
                    } 
                    else
                    {
                        /* wait communication for input activation, forward pass */
                        ActivationImpl* act = op->GetInput(e.entIdx);
                        ActivationImpl* peerAct = act->GetPeer();
                        if (!peerAct) return;
                        OperationImpl* peerOp = peerAct->GetOp();
                        opIdx = peerOp->GetOpIndex();
                        statIdx = peerOp->GetInputCount() + peerAct->GetActIndex();
                    }
                }
            }
            else
            {
                if (!e.isInputOrIncrement) { statIdx = op->GetOutputCount() + op->GetInputCount() + 2 * e.entIdx; }
                else { statIdx = op->GetOutputCount() + op->GetInputCount() + 2 * e.entIdx + 1; };
            }

            if (statsImpl->GetGlobalTime() == 0) statsImpl->SetGlobalTime(rdtsc());

            unsigned long long delta = rdtsc() - statsImpl->GetGlobalTime();
            if (e.isCompTime)
            {
                if (e.actionType == StatEvent::Start) { statsImpl->GetStatData()[opIdx].compStartTimes[statIdx] += delta; }
                else { statsImpl->GetStatData()[opIdx].compWaitTimes[statIdx] += delta; }

                stats[opIdx].compTime += delta;
                totalCompTime += delta;
            }
            else
            {
                if (e.actionType == StatEvent::Start) { statsImpl->GetStatData()[opIdx].commStartTimes[statIdx] += delta; }
                else { statsImpl->GetStatData()[opIdx].commWaitTimes[statIdx] += delta; }

                if (e.actionType == StatEvent::Start) /* increment comm size only for Start call */
                {
                    size_t commSize = statsImpl->GetStatData()[opIdx].isolationCommSizesPerBatch[statIdx];
                    statsImpl->GetStatData()[opIdx].commSizes[statIdx] += commSize;
                    totalCommSize += commSize;
                    stats[opIdx].commSize += commSize;
                }

                stats[opIdx].commTime += delta;
                totalCommTime += delta;
            }

            if (e.isParam &&
                ((e.actionType == StatEvent::Wait) || (e.actionType == StatEvent::Test && stats[opIdx].isTested[statIdx] == false)) &&
                e.isCompTime &&
                !e.isInputOrIncrement &&
                (opIdx == (size_t)statsImpl->GetStartOpMonitor()) &&
                e.entIdx == 0)
            {
                batchCount++;
            }

            if (e.actionType == StatEvent::Test && e.isCompTime)
                stats[opIdx].isTested[statIdx] = true;

            if ((e.actionType == StatEvent::Start) && (opIdx == session->GetOperationCount() - 1))
                    for (size_t opIdx = 0; opIdx < session->GetOperationCount(); opIdx++)
                        std::fill(stats[opIdx].isTested.begin(), stats[opIdx].isTested.end(), false);

            statsImpl->SetGlobalTime(rdtsc());
        }
    }
};
