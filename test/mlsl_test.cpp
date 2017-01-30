/*
 Copyright (C) 2016 Intel Corporation.
 
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
/* The Intel(R) Machine Learning Scaling Library API usage example and correctness check test */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mlsl.h"

using namespace MLSL;

/* Logging stuff */

#define PRINT_BUF_LEN             4096
#define PRINT_BUF_FLUSH_THRESHOLD 3000
char printBuf[PRINT_BUF_LEN];
int printCount=0;

#define MYFLUSH()               \
  do                            \
  {                             \
      printBuf[printCount] = 0; \
      printf("%s", printBuf);   \
      printCount = 0;           \
      fflush(stdout);           \
  } while(0)

#define MYPRINTF(...)                                                                   \
  do                                                                                    \
  {                                                                                     \
      int c = snprintf(printBuf + printCount, PRINT_BUF_LEN - printCount, __VA_ARGS__); \
      if (c > 0 && c < PRINT_BUF_LEN - printCount)                                      \
          printCount += c;                                                              \
      if (printCount > PRINT_BUF_FLUSH_THRESHOLD)                                       \
          MYFLUSH();                                                                    \
  } while(0)

#define MYASSERT(cond,...)                                                    \
  do                                                                          \
  {                                                                           \
      if (!(cond))                                                            \
      {                                                                       \
          printf("%s:%d:assertion '%s' failed\n", __FILE__, __LINE__, #cond); \
          printf(__VA_ARGS__);                                                \
          Finalize();                                                         \
          exit(1);                                                            \
      }                                                                       \
  } while(0)


/* MLSL Test stuff */

#define DTYPE                 float
#define DTYPE_SIZE            sizeof(DTYPE)
#define CACHELINE_SIZE        64
#define FAIL_COUNTER_MAX      5

#define GLOBAL_MINIBATCH_SIZE 16
#define NUM_LAYERS            2
#define NUM_EPOCHS            2
#define MINIBATCH_PER_EPOCH   3

class Layer;

Distribution *globalDistribution;
Layer        *layers[NUM_LAYERS];
ComputeOp    *operations[NUM_LAYERS];

int nodeId;
int numGlobalNodes;
int numGroups = 1;
bool distUpdate = false;

enum LayerType
{
    CONV_MIMO = 0,
    CONV_FLAT = 1,
    FC        = 2
};

struct LayerParams
{
    int layerId;
    LayerType type;
    int ifm;
    int ofm;
    int ifmWidth;
    int ifmHeight;
    int ofmWidth;
    int ofmHeight;
    int kw;
    int kh;
};
LayerParams  layerParams[NUM_LAYERS];

class Layer
{
    int layerId;
    ComputeOp *op;
    DTYPE *inputFmBuf, *outputFmBuf;                    /* input feature map, output feature map */
    DTYPE *inputFmGradBuf, *outputFmGradBuf;            /* gradients wrt input feature map, ouput feature map */
    DTYPE *weightsBuf, *weightsGradBuf, *weightsIncBuf; /* weights, gradient wrt weights, weights increment */
    bool isBackwardUnpackCalled;

public:
    Layer(int layerId, ComputeOp *op, Layer *prevLayer) : layerId(layerId), op(op), isBackwardUnpackCalled(false)
    {
    	MYASSERT(op, "operation is NULL");
        MYASSERT(op->InputFeatureMap(0), "input feature map is NULL");
        MYASSERT(op->GetWeights(0), "weights are NULL");

        int inSize, prevOutSize;

        if (prevLayer == NULL)
            prevOutSize = 0;
        else
        {
            MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");
            prevOutSize = prevLayer->op->OutputFeatureMap(0)->LocalLen()
                          * prevLayer->op->LocalMinibatchLen()
                          * prevLayer->op->OutputFeatureMap(0)->FMSize()
                          * DTYPE_SIZE;
        }

        inSize = op->InputFeatureMap(0)->LocalLen() * op->LocalMinibatchLen() * op->InputFeatureMap(0)->FMSize() * DTYPE_SIZE;
        if (prevOutSize > inSize) inSize = prevOutSize;

        inputFmBuf = (DTYPE *)Alloc(inSize, CACHELINE_SIZE);
        inputFmGradBuf = (DTYPE *)Alloc(inSize, CACHELINE_SIZE);

        if (prevLayer != NULL)
        {
            prevLayer->outputFmBuf = inputFmBuf;
            prevLayer->outputFmGradBuf = inputFmGradBuf;
            op->SetPrev(prevLayer->op, 0, 0);
        }

        weightsBuf = (DTYPE *)Alloc(op->GetWeights(0)->LocalLen()
                                    * op->GetWeights(0)->WTSize()
                                    * DTYPE_SIZE, CACHELINE_SIZE);
        weightsGradBuf = (DTYPE *)Alloc(op->GetWeights(0)->LocalLen()
                                        * op->GetWeights(0)->WTSize()
                                        * DTYPE_SIZE, CACHELINE_SIZE);
        weightsIncBuf = (DTYPE *)Alloc(op->GetWeights(0)->OwnedLen()
                                       * op->GetWeights(0)->WTSize()
                                       * DTYPE_SIZE, CACHELINE_SIZE);

        for (int idx = 0; idx < op->GetWeights(0)->LocalLen() * op->GetWeights(0)->WTSize(); idx++)
            weightsBuf[idx] = idx;
    }

    ~Layer()
    {
        Free(inputFmBuf);
        Free(inputFmGradBuf);
        Free(weightsBuf);
        Free(weightsGradBuf);
        Free(weightsIncBuf);
        delete op;
    }

    void PackBuffer(FeatureMap *fm, DTYPE *commBuf, DTYPE *localBuf)
    {
        int localFmCount = fm->LocalLen();
        for (int blockIdx = 0; blockIdx < fm->NumPackBlocks(); blockIdx++)
        {
            BlockInfo *blockInfo = fm->GetPackBlock(blockIdx);
            int mbCount = blockInfo->MBLen();
            int mbOffset = blockInfo->MBStart();
            int fmCount = blockInfo->FMLen();
            int fmOffset = blockInfo->FMStart();
            int fmSize = blockInfo->FMSize();
            DTYPE* src = localBuf;
            DTYPE* dst = commBuf + blockInfo->BufOffset();
            for (int mbIdx = 0; mbIdx < mbCount; mbIdx++)
                for (int fmIdx = 0; fmIdx < fmCount; fmIdx++)
                    for (int spaceIdx = 0 ; spaceIdx < blockInfo->FMSize(); spaceIdx++)
                        dst[mbIdx * fmCount * fmSize + fmIdx * fmSize + spaceIdx]
                            = src[(mbIdx + mbOffset) * localFmCount * fmSize + (fmIdx + fmOffset) * fmSize + spaceIdx];
        }
    }

    void UnpackBuffer(FeatureMap *fm, DTYPE *commBuf, DTYPE *localBuf)
    {
        int localFmCount = fm->LocalLen();
        for (int blockIdx = 0; blockIdx < fm->NumUnpackBlocks(); blockIdx++)
        {
            BlockInfo *blockInfo = fm->GetUnpackBlock(blockIdx);
            int mbCount = blockInfo->MBLen();
            int mbOffset = blockInfo->MBStart();
            int fmCount = blockInfo->FMLen();
            int fmOffset = blockInfo->FMStart();
            int fmSize = blockInfo->FMSize();
            DTYPE *src = commBuf + blockInfo->BufOffset();
            DTYPE *dst = localBuf;
            for (int mbIdx = 0; mbIdx < mbCount; mbIdx++)
                for (int fmIdx = 0; fmIdx < fmCount; fmIdx++)
                    for (int spaceIdx = 0 ; spaceIdx < blockInfo->FMSize(); spaceIdx++)
                        dst[(mbIdx + mbOffset) * localFmCount * fmSize + (fmIdx + fmOffset) * fmSize + spaceIdx]
                            = src[mbIdx * fmCount * fmSize + fmIdx * fmSize + spaceIdx];
        }
    }

    void ForwardCompute(DTYPE *inputFm, DTYPE *weights, DTYPE *outputFm)
    {
        if (layerId == 0)
        {
            /* Write to output feature map */
            MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");
            int outSize = op->OutputFeatureMap(0)->LocalLen() * op->LocalMinibatchLen() * op->OutputFeatureMap(0)->FMSize();
            for (int idx = 0; idx < outSize; idx++)
                outputFm[idx] = idx;
        }
        else if (layerId == 1)
        {
            /* Check for input feature map*/
            MYASSERT(op->InputFeatureMap(0), "input feature map is NULL");
            int fmLocalCount = op->InputFeatureMap(0)->LocalLen();
            int mbLocalLen = op->LocalMinibatchLen();
            int fmSize = op->InputFeatureMap(0)->FMSize();
            int fmOffset =  op->InputFeatureMap(0)->GlobalOffset();
            int fmGroupSize = op->GetDistribution()->GetFMGroupSize();
            int failCounter = 0;
            for (int mbIdx = 0; mbIdx < mbLocalLen; mbIdx++)
            {
                for (int fmIdx = 0; fmIdx < fmLocalCount; fmIdx++)
                {
                    for (int spaceIdx = 0; spaceIdx < fmSize; spaceIdx++)
                    {
                        DTYPE expected = fmGroupSize * (mbIdx * fmLocalCount * fmSize * fmGroupSize + (fmOffset + fmIdx) * fmSize + spaceIdx);
                        int idx = mbIdx * fmLocalCount * fmSize + fmIdx * fmSize + spaceIdx;
                        if (fabs(inputFm[idx] - expected) > 1.e-4)
                        {
                            if (failCounter < FAIL_COUNTER_MAX)
                                MYPRINTF("[%d] forward input feature map: idx %d: expected %4.0f - received: %4.0f\n", nodeId, idx, expected, inputFm[idx]);
                            failCounter++;
                        }
                    }
                }
            }

            if (failCounter > 0)
                MYPRINTF("[%d] forward input feature map test FAILED num mismatch = %d\n", nodeId, failCounter);
            else
                MYPRINTF("[%d] forward input feature map test PASSED\n", nodeId);
        }

        /* Now check Weights */
        MYASSERT(op->GetWeights(0), "weights are NULL");
        int wSize = op->GetWeights(0)->LocalLen() * op->GetWeights(0)->WTSize();
        int failCounter = 0;
        for (int idx = 0; idx < wSize; idx++)
        {
            if (fabs(weights[idx] - idx) > 1.e-4)
            {
                if (failCounter < FAIL_COUNTER_MAX)
                    MYPRINTF("[%d] forward weights: idx %d: expected %4.0f - received: %4.0f\n",
                             nodeId,
                             idx,
                             (DTYPE)idx,
                             weights[idx]);
                failCounter++;
            }
        }

        if(failCounter > 0)
            MYPRINTF("[%d] forward weights test FAILED num mismatch = %d\n", nodeId, failCounter);
        else
            MYPRINTF("[%d] forward weights test PASSED\n", nodeId);

        MYFLUSH();
    }

    void BackwardCompute1(DTYPE *outputFmGrad, DTYPE *weights, DTYPE *inputFmGrad)
    {
        if (layerId == 0)
        {
            /* Check for inputs */
            MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");
            int fmSize = op->OutputFeatureMap(0)->LocalLen() * op->LocalMinibatchLen() * op->OutputFeatureMap(0)->FMSize();
            int failCounter = 0;
            for (int idx = 0; idx < fmSize; idx++)
            {
                if (fabs(outputFmGrad[idx] - idx) > 1.e-4)
                {
                    if (failCounter < FAIL_COUNTER_MAX)
                        MYPRINTF("[%d] backward output feature map gradient: idx %d: expected %4.0f - received: %4.0f\n",
                                 nodeId,
                                 idx,
                                 (DTYPE)idx,
                                 outputFmGrad[idx]);
                    failCounter++;
                }
            }
            if (failCounter > 0)
                MYPRINTF("[%d] backward output feature map gradient test FAILED num mismatch = %d\n", nodeId, failCounter);
            else
                MYPRINTF("[%d] backward output feature map gradient test PASSED\n", nodeId);
        }
        else if (layerId == 1)
        {
            /* Write to output */
            MYASSERT(op->InputFeatureMap(0), "input feature map is NULL");
            int fmLocalCount = op->InputFeatureMap(0)->LocalLen();
            int mbLocalLen = op->LocalMinibatchLen();
            int fmSize = op->InputFeatureMap(0)->FMSize();
            int fmOffset =  op->InputFeatureMap(0)->GlobalOffset();
            int groupSize = op->GetDistribution()->GetFMGroupSize();
            for (int mbIdx = 0; mbIdx < mbLocalLen; mbIdx++)
                for (int fmIdx = 0; fmIdx < fmLocalCount; fmIdx++)
                    for (int spaceIdx = 0; spaceIdx < fmSize; spaceIdx++)
                    {
                        int idx = mbIdx * fmLocalCount * fmSize + fmIdx * fmSize + spaceIdx;
                        inputFmGrad[idx] = mbIdx * fmLocalCount * fmSize * groupSize + (fmOffset + fmIdx) * fmSize + spaceIdx;
                    }
        }
        MYFLUSH();
    }

    void BackwardCompute2(DTYPE *outputFmGrad, DTYPE *inputFm, DTYPE *weightsGrad)
    {
        MYASSERT(op->GetWeights(0), "weights are NULL");
        int wSize = op->GetWeights(0)->LocalLen() * op->GetWeights(0)->WTSize();
        for (int idx = 0; idx < wSize; idx++)
            weightsGrad[idx] = idx;
    }

    void UpdateCompute(DTYPE *weightsGrad, DTYPE *weightsInc, DTYPE *ownedWeights, int ownedSize)
    {
        MYASSERT(op->GetWeights(0), "weights are NULL");
        int mbGroupSize = op->GetDistribution()->GetMBGroupSize();
        int ownedOffset = op->GetWeights(0)->OwnedStart() * op->GetWeights(0)->WTSize();
        int failCounter = 0;
        for (int idx = 0; idx < ownedSize; idx++)
        {
            DTYPE expected = mbGroupSize * (ownedOffset + idx);
            if (fabs(weightsGrad[idx] - expected) > 1.e-4)
                failCounter++;
            ownedWeights[idx] = ownedOffset + idx;
        }
        if (failCounter > 0)
            MYPRINTF("[%d] weights gradient test FAILED num mismatch = %d\n", nodeId, failCounter);
        else
            MYPRINTF("[%d] weights gradient test PASSED\n", nodeId);
        MYFLUSH();
    }

    /* Recv weights increments (in case of distributed update) and input feature maps, send output feature maps */
    void Forward()
    {
        MYASSERT(op->InputFeatureMap(0), "input feature map is NULL");
        MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");

        FeatureMap *fm = op->InputFeatureMap(0);
        DTYPE *commBuf = (DTYPE *)fm->CommsWait();
        UnpackBuffer(fm, commBuf, inputFmBuf);
        if (op->HasWeights())
        {
            MYASSERT(op->GetWeights(0), "weights are NULL");
            op->GetWeights(0)->CommsWaitWtInc();
        }

        ForwardCompute(inputFmBuf, weightsBuf, outputFmBuf);

        fm = op->OutputFeatureMap(0);
        DTYPE * outputFmCommBuf = (DTYPE *)fm->CBuf()->GetPtr();
        PackBuffer(fm, outputFmCommBuf, outputFmBuf);
        fm->CommsStart(outputFmCommBuf);
        isBackwardUnpackCalled = false;
    }

    /* Calculate gradient wrt input feature mapand send it */
    void Backward1()
    {
        MYASSERT(op->InputFeatureMap(0), "input feature map is NULL");

        if (!isBackwardUnpackCalled)
        {
            MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");
            FeatureMap *fm = op->OutputFeatureMap(0);
            DTYPE *commBuf = (DTYPE *)fm->CommsWait();
            UnpackBuffer(fm, commBuf, outputFmGradBuf);
            isBackwardUnpackCalled = true;
        }

        BackwardCompute1(outputFmGradBuf, weightsBuf, inputFmGradBuf);

        FeatureMap *fm = op->InputFeatureMap(0);
        DTYPE * inputFmCommBuf = (DTYPE *)fm->CBuf()->GetPtr();
        PackBuffer(fm, inputFmCommBuf, inputFmGradBuf);
        fm->CommsStart(inputFmCommBuf);
    }

    /* Calculate gradient wrt weights and send it */
    void Backward2()
    {
        if (!isBackwardUnpackCalled)
        {
            MYASSERT(op->OutputFeatureMap(0), "output feature map is NULL");
            FeatureMap *fm = op->OutputFeatureMap(0);
            DTYPE *commBuf = (DTYPE *)fm->CommsWait();
            UnpackBuffer(fm, commBuf, outputFmGradBuf);
            isBackwardUnpackCalled = true;
        }

        BackwardCompute2(outputFmGradBuf, inputFmBuf, weightsGradBuf);
        if (op->HasWeights())
        {
            MYASSERT(op->GetWeights(0), "weights are NULL");
            op->GetWeights(0)->CommsStartDelWt(weightsGradBuf);
        }
    }

    /* Recv gradient wrt weights and update weights/send weights increments (in case of distributed update) */
    void Update()
    {
        if (op->HasWeights())
        {
            MYASSERT(op->GetWeights(0), "weights are NULL");
            DTYPE *commBuf = (DTYPE*)op->GetWeights(0)->CommsWaitDelWt();
            DTYPE *ownedWeightsBuf = weightsBuf + op->GetWeights(0)->OwnedStart() * op->GetWeights(0)->WTSize();
            UpdateCompute(commBuf == NULL ? weightsGradBuf : commBuf,
                          weightsIncBuf,
                          ownedWeightsBuf,
                          op->GetWeights(0)->OwnedLen() * op->GetWeights(0)->WTSize());
            op->GetWeights(0)->CommsStartWtInc(weightsBuf);
        }
    }
};

/* Layer initialization */
Layer *CreateLayer(LayerType type, LayerParams *lParams, Distribution *distribution, Layer *prevLayer)
{
    MYASSERT((type == CONV_MIMO || type == CONV_FLAT || type == FC), "incorrect op type");

    int layerId = lParams->layerId;

    ComputeOpRegInfo *regInfo = new ComputeOpRegInfo(COMP_OP_TYPE_CC);
    regInfo->SetName("MyLayerName");
    regInfo->AddInputFeatureMap(lParams->ifm, lParams->ifmWidth * lParams->ifmHeight, (DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE);
    regInfo->AddOutputFeatureMap(lParams->ofm, lParams->ofmWidth * lParams->ofmHeight, (DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE);
    regInfo->AddWeights(lParams->ifm * lParams->ofm, lParams->kw * lParams->kh, (DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE, distUpdate);

    ComputeOp *op = new ComputeOp(regInfo, distribution);
    operations[layerId] = op;
    Layer *layer = new Layer(layerId, op, prevLayer);
    delete regInfo;

    return layer;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        printf("specify parameters: mlsl_test [NUM_GROUPS] [DIST_UPDATE]\n");
        return 0;
    }

    int runtime_version = GetVersion();
    printf("built with MLSL API version: %d.%d, used MLSL API version: %d.%d\n",
           MLSL_MAJOR_VERSION, MLSL_MINOR_VERSION, MLSL_MAJOR(runtime_version), MLSL_MINOR(runtime_version));

    if (MLSL_MAJOR_VERSION != MLSL_MAJOR(runtime_version))
    {
        printf("incompatible MLSL API version: %d.%d, exit\n",
               MLSL_MAJOR(runtime_version), MLSL_MINOR(runtime_version));
        return 0;
    }

    Init(&argc, &argv);
    SetMinibatchSize(GLOBAL_MINIBATCH_SIZE);

    numGlobalNodes = GetNumNodes();
    if (argc > 1) numGroups = atoi(argv[1]);
    if (argc > 2) distUpdate = (atoi(argv[2]) != 0);
    if (numGroups < 1) numGroups = 1;
    if (numGroups > numGlobalNodes) numGroups = numGlobalNodes;

    nodeId = GetNodeId();
    if (nodeId == 0)
        printf("num_nodes = %d, distribution = %dx%d (MBParts x FMParts), dist_update = %d\n",
               numGlobalNodes, numGlobalNodes/numGroups, numGroups, distUpdate);

    /* Correctness test assumes both the layers use same distribution */
    globalDistribution = new Distribution(numGlobalNodes/numGroups, numGroups);

    /* Init all the layers */
    int layerIdx;
    for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
    {
        /* Set layerParams for each layer */
        if (layerIdx == 0)
        {
            layerParams[layerIdx].layerId = layerIdx;
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
            layerParams[layerIdx].layerId = layerIdx;
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

        layers[layerIdx] = CreateLayer(layerParams[layerIdx].type,
                                       &layerParams[layerIdx],
                                       globalDistribution,
                                       (layerIdx == 0 ? NULL : layers[layerIdx - 1]));
    }

    for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
    {
        operations[layerIdx]->Finalize();
        operations[layerIdx]->AllocCommsBufs();
    }

    /* Optional Barrier call */
    Barrier();

    for (int epochIdx = 0; epochIdx < NUM_EPOCHS; epochIdx++)
    {
        for (int mbIdx = 0; mbIdx < MINIBATCH_PER_EPOCH; mbIdx++)
        {
            for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
                layers[layerIdx]->Forward();

            for (layerIdx = NUM_LAYERS - 1; layerIdx >= 0; layerIdx--)
            {
                /* Split backward phase on 2 steps to achieve comp/comm overlapping */
                layers[layerIdx]->Backward1();
                layers[layerIdx]->Backward2();
            }

            for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
                layers[layerIdx]->Update();
        }

        /* Finish Weights comms before ending epoch */
        for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
        {
            ComputeOp *op = operations[layerIdx];
            if (op->HasWeights())
            {
                MYASSERT(op->GetWeights(0), "weights are NULL");
                op->GetWeights(0)->CommsWaitWtInc();
            }
        }
    }

    /* Let everybody finish */
    Barrier();

    for (layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
        delete layers[layerIdx];

    delete globalDistribution;

    Finalize();

    printf("[%d] exited normally\n", nodeId);

    return 0;
}
