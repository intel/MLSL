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

/* Intel(R) MLSL library API usage example */

#include <cstdio>  /* printf */
#include <cstdlib> /* atoi */
#include <string.h>

#include "mlsl.hpp"

using namespace MLSL;

#define DTYPE                 float
#define DTYPE_SIZE            sizeof(DTYPE)
#define MLSL_DTYPE            ((DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE)
#define CACHELINE_SIZE        64
#define GLOBAL_MINIBATCH_SIZE 16
#define LAYER_COUNT           2
#define EPOCH_COUNT           10
#define MINIBATCH_PER_EPOCH   10

CompressionType compressType = CT_NONE;
QuantParams* quantParams = NULL;

class Layer
{
    Operation* op;
    DTYPE* inputActBuf;      /* input activation */
    DTYPE* inputActGradBuf;  /* gradients wrt input activation */
    DTYPE* outputActBuf;     /* output activation */
    DTYPE* outputActGradBuf; /* gradients wrt ouput activation */
    DTYPE* paramBuf;         /* learnable parameters */
    DTYPE* paramGradBuf;     /* gradient wrt learnable parameters */

public:
    Layer(Operation* op, Layer* prevLayer) : op(op)
    {
        size_t prevOutActSize = (prevLayer == NULL) ? 0 :
                             prevLayer->op->GetOutput(0)->GetLocalFmCount() * prevLayer->op->GetLocalMinibatchSize()
                             * prevLayer->op->GetOutput(0)->GetFmSize() * DTYPE_SIZE;
        size_t inActSize = op->GetInput(0)->GetLocalFmCount() * op->GetLocalMinibatchSize()
                           * op->GetInput(0)->GetFmSize() * DTYPE_SIZE;

        if (prevOutActSize > inActSize) inActSize = prevOutActSize;
        inputActBuf = (DTYPE*)Environment::GetEnv().Alloc(inActSize, CACHELINE_SIZE);
        inputActGradBuf = (DTYPE*)Environment::GetEnv().Alloc(inActSize, CACHELINE_SIZE);

        if (prevLayer != NULL)
        {
            prevLayer->outputActBuf = inputActBuf;
            prevLayer->outputActGradBuf = inputActGradBuf;
            op->SetPrev(prevLayer->op, 0, 0);
        }

        size_t paramSize = op->GetParameterSet(0)->GetLocalKernelCount()
                           * op->GetParameterSet(0)->GetKernelSize() * DTYPE_SIZE;
        paramBuf = (DTYPE*)Environment::GetEnv().Alloc(paramSize, CACHELINE_SIZE);
        paramGradBuf = (DTYPE*)Environment::GetEnv().Alloc(paramSize, CACHELINE_SIZE);
    }

    ~Layer()
    {
        Environment::GetEnv().Free(inputActBuf);
        Environment::GetEnv().Free(inputActGradBuf);
        Environment::GetEnv().Free(paramBuf);
        Environment::GetEnv().Free(paramGradBuf);
    }

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
                    for (size_t spaceIdx = 0 ; spaceIdx < blockInfo->GetFmSize(); spaceIdx++)
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
                    for (size_t spaceIdx = 0 ; spaceIdx < blockInfo->GetFmSize(); spaceIdx++)
                        dst[(mbIdx + mbOffset) * localFmCount * fmSize + (fmIdx + fmOffset) * fmSize + spaceIdx]
                            = src[mbIdx * fmCount * fmSize + fmIdx * fmSize + spaceIdx];
        }
    }

    void Forward()
    {
        Activation* act = op->GetInput(0);
        DTYPE* commBuf = (DTYPE*)act->WaitComm();
        UnpackBuffer(act, commBuf, inputActBuf);

        /* Layers's forward pass implementation should be here */

        act = op->GetOutput(0);
        DTYPE* outputActCommBuf = (DTYPE*)act->GetCommBuf();
        PackBuffer(act, outputActCommBuf, outputActBuf);
        act->StartComm(outputActCommBuf);
    }

    void Backward()
    {
        Activation* act = op->GetOutput(0);
        DTYPE* commBuf = (DTYPE*)act->WaitComm();
        UnpackBuffer(act, commBuf, outputActGradBuf);

        /* Layers's backward pass implementation should be here */

        act = op->GetInput(0);
        DTYPE* inputActCommBuf = (DTYPE*)act->GetCommBuf();
        PackBuffer(act, inputActCommBuf, inputActGradBuf);
        act->StartComm(inputActCommBuf);

        op->GetParameterSet(0)->StartGradientComm(paramGradBuf);
    }

    void Update()
    {
        DTYPE* commBuf = (DTYPE*)op->GetParameterSet(0)->WaitGradientComm();

        /* Layer's parameters update using optimization algorithm should be here */
    }
};

Layer* CreateLayer(Session* session, Distribution* distribution, Layer* prevLayer)
{
    OperationRegInfo* regInfo = session->CreateOperationRegInfo(OT_CC);
    regInfo->AddInput(256 /* feature map count */, 400 /* feature map size */, MLSL_DTYPE);
    regInfo->AddOutput(256 /* feature map count */, 400 /* feature map size */, MLSL_DTYPE);
    regInfo->AddParameterSet(256 * 256 /* kernel count */, 9 /* kernel size */, MLSL_DTYPE,
                             false /* use dist update */ , compressType);
    size_t opIdx = session->AddOperation(regInfo, distribution);
    session->DeleteOperationRegInfo(regInfo);
    return new Layer(session->GetOperation(opIdx), prevLayer);
}

Layer* layers[LAYER_COUNT];
int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        printf("specify parameters: mlsl_example MODEL_PARTS [PATH_TO_QUANTIZATION_LIB]\n");
        printf("MODEL_PARTS - count of model partitions\n");
        printf("[MODEL_PARTS = 1] - pure data parallelism\n");
        printf("[MODEL_PARTS = N, where N is number of Intel(R) MLSL processes] - pure model parallelism\n");
        printf("[MODEL_PARTS = M, where 1 < M < N] - hybrid parallelism\n");
        printf("[PATH_TO_QUANTIZATION_LIB] - path to quantization library\n");
        return 0;
    }

    Environment::GetEnv().Init(&argc, &argv);
    Session* session = Environment::GetEnv().CreateSession();
    session->SetGlobalMinibatchSize(GLOBAL_MINIBATCH_SIZE);
    size_t processCount = Environment::GetEnv().GetProcessCount();

    size_t modelParts = (size_t)atoi(argv[1]);
    if (modelParts < 1) modelParts = 1;
    if (modelParts > processCount) modelParts = processCount;
    size_t dataParts = processCount/modelParts;
    Distribution* distribution = Environment::GetEnv().CreateDistribution(dataParts, modelParts);

    size_t processIdx = Environment::GetEnv().GetProcessIdx();
    if (processIdx == 0)
        printf("process_count = %zu, distribution = %zu x %zu (data_parts x model_parts)\n",
               processCount, dataParts, modelParts);
    if (argc == 3)
    {
        compressType = CT_QUANTIZATION;
        quantParams = new QuantParams();
        quantParams->lib_path = strdup(argv[2]);
        quantParams->quant_buffer_func_name = strdup("quant_function_name");
        quantParams->dequant_buffer_func_name = strdup("dequant_function_name");
        quantParams->reduce_sum_func_name = strdup("reduce_sum_function_name");
        quantParams->block_size = 268;
        quantParams->elem_in_block = 256;

        Environment::GetEnv().SetQuantizationParams(quantParams);

        free(quantParams->lib_path);
        free(quantParams->quant_buffer_func_name);
        free(quantParams->dequant_buffer_func_name);
        free(quantParams->reduce_sum_func_name);
        delete quantParams;
    }
    size_t layerIdx;
    for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
        layers[layerIdx] = CreateLayer(session, distribution, (layerIdx == 0 ? NULL : layers[layerIdx - 1]));
    session->Commit();

    for (size_t epochIdx = 0; epochIdx < EPOCH_COUNT; epochIdx++)
    {
        for (size_t mbIdx = 0; mbIdx < MINIBATCH_PER_EPOCH; mbIdx++)
        {
            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
                layers[layerIdx]->Forward();

            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
                layers[LAYER_COUNT - layerIdx - 1]->Backward();

            for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
                layers[layerIdx]->Update();
        }
    }

    for (layerIdx = 0; layerIdx < LAYER_COUNT; layerIdx++)
        delete layers[layerIdx];

    Environment::GetEnv().DeleteSession(session);
    Environment::GetEnv().DeleteDistribution(distribution);
    Environment::GetEnv().Finalize();

    printf("[%zu] exited normally\n", processIdx);

    return 0;
}
