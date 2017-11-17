/*
 Copyright (C) 2017 Intel Corporation.
 
 The Intel(R) Machine Learning Scaling Library ("Software") is furnished under
 license and may only be used or copied in accordance with the terms of that
 license. No license, express or implied, by estoppel or otherwise, to any
 intellectual property rights is granted by this document. The Software is
 subject to change without notice, and should not be construed as a commitment
 by Intel Corporation to market, license, sell or support any product or
 technology. Unless otherwise provided for in the license under which this
 Software is provided, the Software is provided AS IS, with no warranties of any
 kind, express or implied. Except as expressly permitted by the Software license,
 neither Intel Corporation nor its suppliers assumes any responsibility or
 liability for any errors or inaccuracies that may appear herein. Except as
 expressly permitted by the Software license, no part of the Software may be
 reproduced, stored in a retrieval system, transmitted in any form, or
 distributed by any means without the express written consent of
 Intel Corporation.
*/

/* MLSL library API usage example and correctness check test */

#include <math.h>   /* fabs */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* exit */

#include <string.h>
#include <sstream>

#include "mlsl.hpp"

using namespace MLSL;
using namespace std;

/* Logging stuff */

#define PRINT_BUF_LEN             4096
#define PRINT_BUF_FLUSH_THRESHOLD 3000
char printBuf[PRINT_BUF_LEN];
int printCount = 0;

#define MY_FLUSH()              \
  do                            \
  {                             \
      printBuf[printCount] = 0; \
      printf("%s", printBuf);   \
      printCount = 0;           \
      fflush(stdout);           \
  } while(0)

#define MY_PRINTF(...)                                                                  \
  do                                                                                    \
  {                                                                                     \
      int c = snprintf(printBuf + printCount, PRINT_BUF_LEN - printCount, __VA_ARGS__); \
      if (c > 0 && c < PRINT_BUF_LEN - printCount)                                      \
          printCount += c;                                                              \
      if (printCount > PRINT_BUF_FLUSH_THRESHOLD)                                       \
          MY_FLUSH();                                                                   \
  } while(0)

#define MY_ASSERT(cond,...)                                                   \
  do                                                                          \
  {                                                                           \
      if (!(cond))                                                            \
      {                                                                       \
          MY_FLUSH();                                                         \
          printf("%s:%d:assertion '%s' failed\n", __FILE__, __LINE__, #cond); \
          printf(__VA_ARGS__);                                                \
          Environment::GetEnv().Finalize();                                   \
          exit(1);                                                            \
      }                                                                       \
  } while(0)


/* MLSL Test stuff */

#define DTYPE                 float
#define DTYPE_SIZE            sizeof(DTYPE)
#define MLSL_DTYPE            ((DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE)
#define CACHELINE_SIZE        64
#define FAIL_COUNTER_MAX      size_t(5)

#define GLOBAL_MINIBATCH_SIZE 16
#define LAYER_COUNT           2
#define EPOCH_COUNT           2
#define MINIBATCH_PER_EPOCH   3

class Layer;

Layer* layers[LAYER_COUNT];
Operation* operations[LAYER_COUNT];

size_t processIdx;
size_t processCount;

/* default parameters */
size_t groupCount = 1;
bool useDistUpdate = false; // data parallelism's feature
bool useUserBuf = false;
bool useTest = false;

QuantParams* quantParams = NULL;
CompressionType compressType = CT_NONE;

enum LayerType
{
    CONV_MIMO = 0,
    CONV_FLAT = 1,
    FC        = 2
};

struct LayerParams
{
    size_t layerIdx;
    LayerType type;
    size_t ifm;
    size_t ofm;
    size_t ifmWidth;
    size_t ifmHeight;
    size_t ofmWidth;
    size_t ofmHeight;
    size_t kw;
    size_t kh;
};
LayerParams layerParams[LAYER_COUNT];

class Layer
{
    size_t layerIdx;
    Operation* op;
    DTYPE* inputActBuf, *outputActBuf;            /* input/output activations */
    DTYPE* inputActGradBuf, *outputActGradBuf;    /* gradients wrt input activations/ouput activations */
    DTYPE* paramBuf, *paramGradBuf, *paramIncBuf;  /* learnable parameters, gradient wrt parameters, parameters increment */
    size_t paramBufCount;
    bool isBackwardUnpackCalled;

public:
    Layer(size_t layerIdx, Operation* op, Layer* prevLayer) : layerIdx(layerIdx), op(op), isBackwardUnpackCalled(false)
    {
        MY_ASSERT(op->GetInput(0), "input activation is NULL");
        MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");

        size_t inActSize, prevOutActSize;

        if (prevLayer == NULL)
            prevOutActSize = 0;
        else
        {
            MY_ASSERT(op->GetOutput(0), "output activation is NULL");
            prevOutActSize = prevLayer->op->GetOutput(0)->GetLocalFmCount()
                             * prevLayer->op->GetLocalMinibatchSize()
                             * prevLayer->op->GetOutput(0)->GetFmSize()
                             * DTYPE_SIZE;
        }

        inActSize = op->GetInput(0)->GetLocalFmCount() * op->GetLocalMinibatchSize() * op->GetInput(0)->GetFmSize() * DTYPE_SIZE;
        if (prevOutActSize > inActSize) inActSize = prevOutActSize;

        inputActBuf = (DTYPE*)malloc(inActSize);
        inputActGradBuf = (DTYPE*)malloc(inActSize);

        if (prevLayer != NULL)
        {
            prevLayer->outputActBuf = inputActBuf;
            prevLayer->outputActGradBuf = inputActGradBuf;
            op->SetPrev(prevLayer->op, 0, 0);
        }

        paramBufCount = op->GetParameterSet(0)->GetLocalKernelCount() * op->GetParameterSet(0)->GetKernelSize();
        size_t paramBufSize = paramBufCount * DTYPE_SIZE;

        size_t paramBufIncSize = op->GetParameterSet(0)->GetOwnedKernelCount()
                                 * op->GetParameterSet(0)->GetKernelSize()
                                 * DTYPE_SIZE;

        if (useUserBuf)
        {
            paramBuf     = (DTYPE*)malloc(paramBufSize);
            paramGradBuf = (DTYPE*)malloc(paramBufSize);
            paramIncBuf  = (DTYPE*)malloc(paramBufIncSize);
        }
        else
        {
            paramBuf     = (DTYPE*)Environment::GetEnv().Alloc(paramBufSize, CACHELINE_SIZE);
            paramGradBuf = (DTYPE*)Environment::GetEnv().Alloc(paramBufSize, CACHELINE_SIZE);
            paramIncBuf  = (DTYPE*)Environment::GetEnv().Alloc(paramBufIncSize, CACHELINE_SIZE);
        }

        MY_ASSERT(inputActBuf && inputActGradBuf && paramBuf && paramGradBuf && paramIncBuf,
                  "error while buffers allocating");

        for (size_t idx = 0; idx < paramBufCount; idx++)
            paramBuf[idx] = idx;
    }

    ~Layer()
    {
        free(inputActBuf);
        free(inputActGradBuf);

        if (useUserBuf)
        {
            free(paramBuf);
            free(paramGradBuf);
            free(paramIncBuf);
        }
        else
        {
            Environment::GetEnv().Free(paramBuf);
            Environment::GetEnv().Free(paramGradBuf);
            Environment::GetEnv().Free(paramIncBuf);
        }
    }

    size_t GetParamBufCount() { return paramBufCount; }
    DTYPE* GetParamBuf() { return paramBuf; }

    void PackBuffer(Activation* act, DTYPE* commBuf, DTYPE* localBuf)
    {
        size_t localFmCount = act->GetLocalFmCount();
        for (size_t blockIdx = 0; blockIdx < act->GetPackBlockCount(); blockIdx++)
        {
            CommBlockInfo* blockInfo = act->GetPackBlock(blockIdx);
            size_t mbCount = blockInfo->GetMbCount();
            size_t mbOffset = blockInfo->GetMbOffset();
            size_t fmCount = blockInfo->GetFmCount();
            size_t fmOffset = blockInfo->GetFmOffset();
            size_t fmSize = blockInfo->GetFmSize();
            DTYPE* src = localBuf;
            DTYPE* dst = commBuf + blockInfo->GetBufOffset();
            for (size_t mbIdx = 0; mbIdx < mbCount; mbIdx++)
                for (size_t fmIdx = 0; fmIdx < fmCount; fmIdx++)
                    for (size_t spaceIdx = 0 ; spaceIdx < fmSize; spaceIdx++)
                        dst[mbIdx * fmCount * fmSize + fmIdx * fmSize + spaceIdx]
                            = src[(mbIdx + mbOffset) * localFmCount * fmSize + (fmIdx + fmOffset) * fmSize + spaceIdx];
        }
    }

    void UnpackBuffer(Activation* act, DTYPE* commBuf, DTYPE* localBuf)
    {
        size_t localFmCount = act->GetLocalFmCount();
        for (size_t blockIdx = 0; blockIdx < act->GetUnpackBlockCount(); blockIdx++)
        {
            CommBlockInfo* blockInfo = act->GetUnpackBlock(blockIdx);
            size_t mbCount = blockInfo->GetMbCount();
            size_t mbOffset = blockInfo->GetMbOffset();
            size_t fmCount = blockInfo->GetFmCount();
            size_t fmOffset = blockInfo->GetFmOffset();
            size_t fmSize = blockInfo->GetFmSize();
            DTYPE* src = commBuf + blockInfo->GetBufOffset();
            DTYPE* dst = localBuf;
            for (size_t mbIdx = 0; mbIdx < mbCount; mbIdx++)
                for (size_t fmIdx = 0; fmIdx < fmCount; fmIdx++)
                    for (size_t spaceIdx = 0 ; spaceIdx < fmSize; spaceIdx++)
                        dst[(mbIdx + mbOffset) * localFmCount * fmSize + (fmIdx + fmOffset) * fmSize + spaceIdx]
                            = src[mbIdx * fmCount * fmSize + fmIdx * fmSize + spaceIdx];
        }
    }

    void ForwardCompute(DTYPE* inputAct, DTYPE* param, DTYPE* outputAct)
    {
        if (layerIdx == 0)
        {
            /* Write to output activation */
            MY_ASSERT(op->GetOutput(0), "output activation is NULL");
            size_t outSize = op->GetOutput(0)->GetLocalFmCount() * op->GetLocalMinibatchSize() * op->GetOutput(0)->GetFmSize();
            for (size_t idx = 0; idx < outSize; idx++)
                outputAct[idx] = idx;
        }
        else if (layerIdx == 1)
        {
            /* Check for input activation */
            MY_ASSERT(op->GetInput(0), "input activation is NULL");
            size_t fmLocalCount = op->GetInput(0)->GetLocalFmCount();
            size_t mbLocalLen = op->GetLocalMinibatchSize();
            size_t fmSize = op->GetInput(0)->GetFmSize();
            size_t fmOffset =  op->GetInput(0)->GetGlobalFmOffset();
            size_t fmGroupSize = op->GetDistribution()->GetProcessCount(GT_MODEL);
            size_t failCounter = 0;
            for (size_t mbIdx = 0; mbIdx < mbLocalLen; mbIdx++)
            {
                for (size_t fmIdx = 0; fmIdx < fmLocalCount; fmIdx++)
                {
                    for (size_t spaceIdx = 0; spaceIdx < fmSize; spaceIdx++)
                    {
                        DTYPE expected = fmGroupSize * (mbIdx * fmLocalCount * fmSize * fmGroupSize + (fmOffset + fmIdx) * fmSize + spaceIdx);
                        size_t idx = mbIdx * fmLocalCount * fmSize + fmIdx * fmSize + spaceIdx;
                        if (fabs(inputAct[idx] - expected) > 1.e-4)
                        {
                            if (failCounter < FAIL_COUNTER_MAX)
                                MY_PRINTF("[%zu] forward_%zu: input: idx %zu: expected %4.0f - received: %4.0f\n",
                                          processIdx, layerIdx, idx, expected, inputAct[idx]);
                            failCounter++;
                        }
                    }
                }
            }

            if (failCounter > 0)
            {
                MY_PRINTF("[%zu] forward_%zu: input activation test FAILED mismatch count = %zu\n", processIdx, layerIdx, failCounter);
                MY_ASSERT(0, "exit");
            }
            else
                MY_PRINTF("[%zu] forward_%zu: input activation test PASSED\n", processIdx, layerIdx);
        }

        /* Now check ParameterSet */
        MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
        size_t paramSize = op->GetParameterSet(0)->GetLocalKernelCount() * op->GetParameterSet(0)->GetKernelSize();
        size_t failCounter = 0;
        for (size_t idx = 0; idx < paramSize; idx++)
        {
            if (fabs(param[idx] - idx) > 1.e-4)
            {
                if (failCounter < FAIL_COUNTER_MAX)
                    MY_PRINTF("[%zu] forward_%zu: parameter idx %zu: expected %4.0f - received: %4.0f\n",
                             processIdx,
                             layerIdx,
                             idx,
                             (DTYPE)idx,
                             param[idx]);
                failCounter++;
            }
        }

        if (failCounter > 0)
        {
            MY_PRINTF("[%zu] forward_%zu: parameter test FAILED mismatch count = %zu\n", processIdx, layerIdx, failCounter);
            MY_ASSERT(0, "exit");
        }
        else
            MY_PRINTF("[%zu] forward_%zu: parameter test PASSED\n", processIdx, layerIdx);
        MY_FLUSH();
    }

    void BackwardCompute1(DTYPE* outputActGrad, DTYPE* param, DTYPE* inputActGrad)
    {
        if (layerIdx == 0)
        {
            /* Check for inputs */
            MY_ASSERT(op->GetOutput(0), "output activation is NULL");
            size_t actSize = op->GetOutput(0)->GetLocalFmCount() * op->GetLocalMinibatchSize() * op->GetOutput(0)->GetFmSize();
            size_t failCounter = 0;
            for (size_t idx = 0; idx < actSize; idx++)
            {
                if (fabs(outputActGrad[idx] - idx) > 1.e-4)
                {
                    if (failCounter < FAIL_COUNTER_MAX)
                        MY_PRINTF("[%zu] backward_%zu: output activation gradient: idx %zu: expected %4.0f - received: %4.0f\n",
                                 processIdx,
                                 layerIdx,
                                 idx,
                                 (DTYPE)idx,
                                 outputActGrad[idx]);
                    failCounter++;
                }
            }
            if (failCounter > 0)
            {
                MY_PRINTF("[%zu] backward_%zu: output activation gradient test FAILED mismatch count = %zu\n", processIdx, layerIdx, failCounter);
                MY_ASSERT(0, "exit");
            }
            else
                MY_PRINTF("[%zu] backward_%zu: output activation gradient test PASSED\n", processIdx, layerIdx);
        }
        else if (layerIdx == 1)
        {
            /* Write to output */
            MY_ASSERT(op->GetInput(0), "input activation is NULL");
            size_t fmLocalCount = op->GetInput(0)->GetLocalFmCount();
            size_t mbLocalLen = op->GetLocalMinibatchSize();
            size_t fmSize = op->GetInput(0)->GetFmSize();
            size_t actOffset =  op->GetInput(0)->GetGlobalFmOffset();
            size_t groupSize = op->GetDistribution()->GetProcessCount(GT_MODEL);
            for (size_t mbIdx = 0; mbIdx < mbLocalLen; mbIdx++)
                for (size_t fmIdx = 0; fmIdx < fmLocalCount; fmIdx++)
                    for (size_t spaceIdx = 0; spaceIdx < fmSize; spaceIdx++)
                    {
                        size_t idx = mbIdx * fmLocalCount * fmSize + fmIdx * fmSize + spaceIdx;
                        inputActGrad[idx] = mbIdx * fmLocalCount * fmSize * groupSize + (actOffset + fmIdx) * fmSize + spaceIdx;
                    }
        }
        MY_FLUSH();
    }

    void BackwardCompute2(DTYPE* outputActGrad, DTYPE* inputAct, DTYPE* paramGrad)
    {
        MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
        size_t paramSize = op->GetParameterSet(0)->GetLocalKernelCount() * op->GetParameterSet(0)->GetKernelSize();
        for (size_t idx = 0; idx < paramSize; idx++)
            paramGrad[idx] = idx;
    }

    void UpdateCompute(DTYPE* paramGrad, DTYPE* paramInc, DTYPE* ownedParam, size_t ownedSize)
    {
        MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
        size_t mbGroupSize = op->GetDistribution()->GetProcessCount(GT_DATA);
        size_t ownedOffset = op->GetParameterSet(0)->GetOwnedKernelOffset() * op->GetParameterSet(0)->GetKernelSize();
        size_t failCounter = 0;
        if (compressType == CT_NONE)
        {
            for (size_t idx = 0; idx < ownedSize; idx++)
            {
                DTYPE expected = mbGroupSize * (ownedOffset + idx);
                if (fabs(paramGrad[idx] - expected) > 1.e-4)
                    failCounter++;
                ownedParam[idx] = ownedOffset + idx;
            }
        }
        else
        {
            DTYPE expected = mbGroupSize * ownedOffset;
            DTYPE diff= fabs(paramGrad[0] - expected);
            if (expected != 0)
                diff /= expected;
            DTYPE min = diff ;
            DTYPE max = diff;
            DTYPE avr = diff;
            size_t diffCount = 0;
            for (size_t idx = 1; idx < ownedSize; idx++)
            {
                expected = mbGroupSize * (ownedOffset + idx);
                diff = fabs(paramGrad[idx] - expected);
                if (expected != 0)
                    diff /= expected;
                if ( min > diff)
                    min = diff;
                if ( max < diff)
                    max = diff;
                if (diff != 0)
                    diffCount++;
                ownedParam[idx] = ownedOffset + idx;
            }
//            MY_PRINTF("[%zu] update_%zu: parameter gradient test. avr diff(\%) = %4.4f\%, min = %4.4f\%, max = %4.4f\%, count different (all) = %zu (%zu)\n",
//                      processIdx, layerIdx, avr / ownedSize * 100.0f, min * 100.0f, max * 100.0f, diffCount, ownedSize);

        }
        if (failCounter > 0)
        {
            MY_PRINTF("[%zu] update_%zu: parameter gradient test FAILED mismatch count = %zu\n", processIdx, layerIdx, failCounter);
            MY_ASSERT(0, "exit");
        }
        else
            MY_PRINTF("[%zu] update_%zu: parameter gradient test PASSED\n", processIdx, layerIdx);
        MY_FLUSH();
    }

    /* Recv parameter increments (in case of distributed update) and input activations, send output activations */
    void Forward()
    {
        MY_ASSERT(op->GetInput(0), "input activation is NULL");
        MY_ASSERT(op->GetOutput(0), "otput activation is NULL");

        Activation* act = op->GetInput(0);
        DTYPE* commBuf = (DTYPE*)act->WaitComm();
        UnpackBuffer(act, commBuf, inputActBuf);
        if (op->HasParameterSets())
        {
            MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
            op->GetParameterSet(0)->WaitIncrementComm();
        }

        ForwardCompute(inputActBuf, paramBuf, outputActBuf);

        act = op->GetOutput(0);
        DTYPE* outputActCommBuf = (DTYPE*)act->GetCommBuf();
        PackBuffer(act, outputActCommBuf, outputActBuf);
        act->StartComm(outputActCommBuf);
        isBackwardUnpackCalled = false;
    }

    /* Calculate gradient wrt input activation and send it */
    void Backward1()
    {
        MY_ASSERT(op->GetInput(0), "input activation is NULL");

        if (!isBackwardUnpackCalled)
        {
            MY_ASSERT(op->GetOutput(0), "output activation is NULL");
            Activation* act = op->GetOutput(0);
            DTYPE* commBuf = (DTYPE*)act->WaitComm();
            UnpackBuffer(act, commBuf, outputActGradBuf);
            isBackwardUnpackCalled = true;
        }

        BackwardCompute1(outputActGradBuf, paramBuf, inputActGradBuf);

        Activation* act = op->GetInput(0);
        DTYPE* inputActCommBuf = (DTYPE*)act->GetCommBuf();
        PackBuffer(act, inputActCommBuf, inputActGradBuf);
        act->StartComm(inputActCommBuf);
    }

    /* Calculate gradient wrt parameters and send it */
    void Backward2()
    {
        if (!isBackwardUnpackCalled)
        {
            MY_ASSERT(op->GetOutput(0), "output activation is NULL");
            Activation* act = op->GetOutput(0);
            DTYPE* commBuf = (DTYPE*)act->WaitComm();
            UnpackBuffer(act, commBuf, outputActGradBuf);
            isBackwardUnpackCalled = true;
        }

        BackwardCompute2(outputActGradBuf, inputActBuf, paramGradBuf);
        if (op->HasParameterSets())
        {
            MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
            op->GetParameterSet(0)->StartGradientComm(paramGradBuf);
        }
    }

    /* Recv gradient wrt parameters and update parameters/send parameter increments (in case of distributed update) */
    void Update()
    {
        if (op->HasParameterSets())
        {
            MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
            DTYPE* commBuf = NULL;
            if (useTest)
            {
                bool isCompleted = false;
                while (!isCompleted)
                    commBuf = (DTYPE*)op->GetParameterSet(0)->TestGradientComm(&isCompleted);
            }
            else
                commBuf = (DTYPE*)op->GetParameterSet(0)->WaitGradientComm();

            DTYPE* ownedParamBuf = paramBuf + op->GetParameterSet(0)->GetOwnedKernelOffset() * op->GetParameterSet(0)->GetKernelSize();
            UpdateCompute(commBuf == NULL ? paramGradBuf : commBuf,
                          paramIncBuf,
                          ownedParamBuf,
                          op->GetParameterSet(0)->GetOwnedKernelCount() * op->GetParameterSet(0)->GetKernelSize());
            op->GetParameterSet(0)->StartIncrementComm(paramBuf);
        }
    }
};

/* Layer initialization */
Layer* CreateLayer(Session* session, LayerType type, LayerParams* lParams, Distribution* distribution, Layer* prevLayer)
{
    MY_ASSERT((type == CONV_MIMO || type == CONV_FLAT || type == FC), "incorrect op type");

    size_t layerIdx = lParams->layerIdx;

    OperationRegInfo* regInfo = session->CreateOperationRegInfo(OT_CC);
    stringstream stream;
    stream << "layer_" << layerIdx;
    regInfo->SetName(stream.str().c_str());
    regInfo->AddInput(lParams->ifm, lParams->ifmWidth * lParams->ifmHeight, MLSL_DTYPE);
    regInfo->AddOutput(lParams->ofm, lParams->ofmWidth * lParams->ofmHeight, MLSL_DTYPE);
    if (compressType == CT_QUANTIZATION)
        regInfo->AddParameterSet(lParams->ifm * lParams->ofm, lParams->kw * lParams->kh, MLSL_DTYPE, useDistUpdate, compressType);
    else
        regInfo->AddParameterSet(lParams->ifm * lParams->ofm, lParams->kw * lParams->kh, MLSL_DTYPE, useDistUpdate);

    size_t opIdx = session->AddOperation(regInfo, distribution);
    session->DeleteOperationRegInfo(regInfo);

    Operation* op = session->GetOperation(opIdx);
    operations[layerIdx] = op;
    Layer* layer = new Layer(layerIdx, op, prevLayer);

    return layer;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("specify parameters: mlsl_test GROUP_COUNT [DIST_UPDATE] [USER_BUF] [USE_TEST] [PATH_TO_QUANTIZATION_LIB]\n");
        exit(0);
    }

    int runtime_version = Environment::GetEnv().GetVersion();
    printf("built with MLSL API version: %d.%d, used MLSL API version: %d.%d\n",
           MLSL_MAJOR_VERSION, MLSL_MINOR_VERSION, MLSL_MAJOR(runtime_version), MLSL_MINOR(runtime_version));

    if (MLSL_MAJOR_VERSION != MLSL_MAJOR(runtime_version))
    {
        printf("incompatible MLSL API version: %d.%d, exit\n",
               MLSL_MAJOR(runtime_version), MLSL_MINOR(runtime_version));
        return 0;
    }

    Environment::GetEnv().Init(&argc, &argv);
    Session* session = Environment::GetEnv().CreateSession();
    session->SetGlobalMinibatchSize(GLOBAL_MINIBATCH_SIZE);
    processCount = Environment::GetEnv().GetProcessCount();

    if (argc > 1) groupCount    = atoi(argv[1]);
    if (argc > 2) useDistUpdate = (atoi(argv[2]) != 0);
    if (argc > 3) useUserBuf    = (atoi(argv[3]) != 0);
    if (argc > 4) useTest       = (atoi(argv[4]) != 0);
    if (argc > 5)
    {
        compressType = CT_QUANTIZATION;
        quantParams = new QuantParams();
        quantParams->lib_path = strdup(argv[5]);
        quantParams->quant_buffer_func_name = strdup("dl_comp_compress_buffer");
        quantParams->dequant_buffer_func_name = strdup("dl_comp_decompress_buffer");
        quantParams->reduce_sum_func_name = strdup("dl_comp_compressed_buffer_reduce_sum");
        quantParams->block_size = 268;
        quantParams->elem_in_block = 256;
        Environment::GetEnv().SetQuantizationParams(quantParams);
        free(quantParams->lib_path);
        free(quantParams->quant_buffer_func_name);
        free(quantParams->dequant_buffer_func_name);
        free(quantParams->reduce_sum_func_name);
        delete quantParams;
    }

    if (groupCount < 1) groupCount = 1;
    if (groupCount > processCount) groupCount = processCount;

    processIdx = Environment::GetEnv().GetProcessIdx();
    if (processIdx == 0)
        printf("\nprocess_count = %zu, distribution = %zu x %zu (data_parts x model_parts), dist_update %d, user_buf %d, use_test %d\n\n",
               processCount, processCount/groupCount, groupCount, useDistUpdate, useUserBuf, useTest);

    /* Correctness test assumes both the layers use same distribution */
    Distribution* distribution = Environment::GetEnv().CreateDistribution(processCount/groupCount, groupCount);

    /* Init all the layers */
    size_t layerIdx;
    for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
    {
        /* Set layerParams for each layer */
        if (layerIdx == 0)
        {
            layerParams[layerIdx].layerIdx = layerIdx;
            layerParams[layerIdx].type = CONV_MIMO;
            layerParams[layerIdx].ifm  = 128;
            layerParams[layerIdx].ofm  = 256;
            layerParams[layerIdx].ifmWidth = 12;
            layerParams[layerIdx].ifmHeight = 12;
            layerParams[layerIdx].ofmWidth = 12;
            layerParams[layerIdx].ofmHeight = 12;
            layerParams[layerIdx].kw = 3;
            layerParams[layerIdx].kh = 3;
        }
        else if (layerIdx == 1)
        {
            layerParams[layerIdx].layerIdx = layerIdx;
            layerParams[layerIdx].type = CONV_MIMO;
            layerParams[layerIdx].ifm  = 256;
            layerParams[layerIdx].ofm  = 256;
            layerParams[layerIdx].ifmWidth = 12;
            layerParams[layerIdx].ifmHeight = 12;
            layerParams[layerIdx].ofmWidth = 12;
            layerParams[layerIdx].ofmHeight = 12;
            layerParams[layerIdx].kw = 3;
            layerParams[layerIdx].kh = 3;
        }

        layers[layerIdx] = CreateLayer(session,
                                       layerParams[layerIdx].type,
                                       &layerParams[layerIdx],
                                       distribution,
                                       (layerIdx == 0 ? NULL : layers[layerIdx - 1]));
        CommReq* req = distribution->Bcast(layers[layerIdx]->GetParamBuf(), layers[layerIdx]->GetParamBufCount(), MLSL_DTYPE, 0, GT_GLOBAL);
        Environment::GetEnv().Wait(req);
    }

    session->Commit();

    Statistics* stats = session->GetStats();
    stats->Start();

    for (size_t epochIdx = 0; epochIdx < EPOCH_COUNT; epochIdx++)
    {
        for (size_t mbIdx = 0; mbIdx < MINIBATCH_PER_EPOCH; mbIdx++)
        {
            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
                layers[layerIdx]->Forward();

            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
            {
                /* Split backward phase on 2 steps to achieve comp/comm overlapping */
                layers[LAYER_COUNT - layerIdx - 1]->Backward1();
                layers[LAYER_COUNT - layerIdx - 1]->Backward2();
            }

            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
            {
                layers[layerIdx]->Update();
            }

            if (stats->IsEnabled())
            {
                printf("\n ++[%zu] total isolation comm cycles: %llu", processIdx, stats->GetTotalIsolationCommCycles());
                printf("\n ++[%zu] total communication bytes: %zu", processIdx, stats->GetTotalCommSize());
                printf("\n ++[%zu] total communication cycles: %lld", processIdx, stats->GetTotalCommCycles());
                printf("\n ++[%zu] total compute cycles: %llu\n", processIdx, stats->GetTotalComputeCycles());
            }
        }

        /* Finish ParameterSet comms before ending epoch */
        for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
        {
            Operation* op = operations[layerIdx];
            if (op->HasParameterSets())
            {
                MY_ASSERT(op->GetParameterSet(0), "parameter is NULL");
                op->GetParameterSet(0)->WaitIncrementComm();
            }
        }
    }

    stats->Stop();
    stats->Print();

    for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
        delete layers[layerIdx];

    Environment::GetEnv().DeleteSession(session);
    Environment::GetEnv().DeleteDistribution(distribution);
    Environment::GetEnv().Finalize();

    printf("[%zu] exited normally\n", processIdx);

    return 0;
}
