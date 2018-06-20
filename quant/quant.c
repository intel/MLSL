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
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "quant.h"
#include "memory.h"
#include "uthash.h"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define GET_TID()    syscall(SYS_gettid)

#define ASSERT(cond, fmt, ...)                                                            \
  do                                                                                      \
  {                                                                                       \
      if (!(cond))                                                                        \
      {                                                                                   \
          fprintf(stderr, "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n",               \
                  GET_TID(), __FILENAME__, __FUNCTION__, __LINE__, #cond, ##__VA_ARGS__); \
          fflush(stderr);                                                                 \
          MPI_Finalize();                                                                 \
          _exit(1);                                                                       \
      }                                                                                   \
  } while(0)

typedef enum
{
    DL_COMP_NONE = 0,
    DL_COMP_DFP = 1,
} dl_comp_method_t;

typedef enum
{
    DL_COMP_INT8    = 0,
    DL_COMP_FLOAT16 = 1,
    DL_COMP_FLOAT32 = 2,
    DL_COMP_FLOAT64 = 3,
} dl_comp_data_type_t;

typedef int (*quant_func_t)(void* src_buffer,
                            void* dst_buffer,
                            size_t count,
                            void* diff,
                            dl_comp_data_type_t src_data_type,
                            size_t comp_ratio,
                            dl_comp_method_t method);
typedef int (*dequant_func_t)(void* src_buffer, void* dst_buffer, size_t count);
typedef int (*reduce_sum_func_t)(const void* in_buffer, void* inout_buffer, size_t blockCount);

typedef struct
{
    quant_func_t quant_func;
    dequant_func_t dequant_func;
    void* quant_lib;
    MPI_Op quant_op;
    size_t block_size;    /* quantize meta data: one block's bytes */
    size_t elem_in_block; /* quantize meta data: how many elements in one block */
} quant_context_t;

typedef struct
{
    void* buf;
    void* diff;
    UT_hash_handle hh;
} quant_diff_map;

quant_diff_map* quant_diff_head = NULL;
MPI_Datatype quant_dt;
reduce_sum_func_t reduce_sum_func;
quant_context_t* qcontext = NULL;

void reduce_sum_func_wrapper(void* invec, void* inoutvec, int* len, MPI_Datatype* dt)
{
    ASSERT(*dt == quant_dt, "unexpected datatype %ld", (long)*dt);
    int ret = reduce_sum_func(invec, inoutvec, *len);
    ASSERT(ret == 0, "reduce failed: error code %d", ret);
}

void quant_load(quant_params_t* qparam)
{
    ASSERT(qparam, "quantization parameters are not set");

    qcontext = (quant_context_t*)malloc(sizeof(quant_context_t));
    ASSERT(qcontext, "memory can't be allocated");
    qcontext->dequant_func = NULL;
    qcontext->quant_op = MPI_SUM;
    qcontext->quant_func = NULL;
    qcontext->elem_in_block = 0;

    setenv("OMP_NUM_THREADS", "1", 1);

    ASSERT(qparam->lib_path, "path to quantization library is not set");
    ASSERT(strcmp(qparam->lib_path, "") != 0, "path to quantization library is empty");

    qcontext->quant_lib = dlopen(qparam->lib_path, RTLD_NOW);
    ASSERT(qcontext->quant_lib, "quantization library can't be opened %s", qparam->lib_path);

    qcontext->quant_func = dlsym(qcontext->quant_lib, qparam->quant_buffer_func_name);
    ASSERT(qcontext->quant_func, "quantization function can't be loaded %s", qparam->quant_buffer_func_name);

    qcontext->dequant_func = dlsym(qcontext->quant_lib, qparam->dequant_buffer_func_name);
    ASSERT(qcontext->dequant_func, "dequantization function can't be loaded %s", qparam->dequant_buffer_func_name);

    reduce_sum_func = dlsym(qcontext->quant_lib, qparam->reduce_sum_func_name);
    ASSERT(reduce_sum_func, "reduce function can't be loaded %s", qparam->reduce_sum_func_name);

    MPI_Op_create(reduce_sum_func_wrapper, 1, &(qcontext->quant_op));
    int block_lengths[1] = { 0 };
    block_lengths[0] = qparam->block_size;
    qcontext->block_size = qparam->block_size;
    qcontext->elem_in_block = qparam->elem_in_block;
    MPI_Aint block_offsets[] = { 0 };
    MPI_Datatype block_types[] = { MPI_CHAR };
    MPI_Type_create_struct(1, block_lengths, block_offsets, block_types, &quant_dt);
    MPI_Type_commit(&quant_dt);
}

size_t quant_get_reduce_count(size_t count)
{
    ASSERT(qcontext != NULL, "quantization library is not loaded");
    return (count + (qcontext->elem_in_block - 1)) / qcontext->elem_in_block;
}

MPI_Op quant_get_op()
{
    ASSERT(qcontext != NULL, "quantization library is not loaded");
    return qcontext->quant_op;
}

MPI_Datatype quant_get_data_type()
{
    ASSERT(qcontext != NULL, "quantization library is not loaded");
    return quant_dt;
}

void* quant_get_diff(void* buf, int count)
{
    quant_diff_map* q_diff;
    HASH_FIND_PTR(quant_diff_head, &buf, q_diff);
    if (!q_diff)
    {
        q_diff = (quant_diff_map*)malloc(sizeof(quant_diff_map));
        ASSERT(q_diff, "memory can't be allocated");
        q_diff->buf = buf;
        q_diff->diff = calloc(count, sizeof(float));
        ASSERT(q_diff->diff, "memory can't be allocated");
        HASH_ADD_PTR(quant_diff_head, buf, q_diff);
    }
    return q_diff->diff;
}

void quant_free_diffs()
{
    quant_diff_map* q_diff_current;
    quant_diff_map* q_diff_tmp;
    HASH_ITER(hh, quant_diff_head, q_diff_current, q_diff_tmp)
    {
        HASH_DEL(quant_diff_head, q_diff_current);
        free(q_diff_current->diff);
        free(q_diff_current);
    }
}

void quant_init(quant_params_t* ptr_qparam)
{
    ASSERT(qcontext == NULL, "quantization library can be initialized only once");
    ASSERT(ptr_qparam != NULL, "quantization parameters are not set");
    quant_params_t qparam;
    
    qparam.lib_path = (char*)memory_translate_clientaddr(ptr_qparam->lib_path);
    qparam.quant_buffer_func_name = (char*)memory_translate_clientaddr(ptr_qparam->quant_buffer_func_name);
    qparam.dequant_buffer_func_name = (char*)memory_translate_clientaddr(ptr_qparam->dequant_buffer_func_name);
    qparam.reduce_sum_func_name = (char*)memory_translate_clientaddr(ptr_qparam->reduce_sum_func_name);
    qparam.block_size = ptr_qparam->block_size;
    qparam.elem_in_block = ptr_qparam->elem_in_block;

    ASSERT(qparam.lib_path != NULL, "quantization library name is not set");
    quant_load(&qparam);
}

void quant_quantize(void* buf, size_t count)
{
    ASSERT(qcontext != NULL, "quantization library is not loaded");
    int ret = qcontext->quant_func(buf, buf, count, quant_get_diff(buf, count),
                                   DL_COMP_FLOAT32, 4, DL_COMP_DFP);
    ASSERT(ret == 0, "quantization failed: error code %d", ret);
}

void quant_dequantize(void* buf, size_t count)
{
    ASSERT(qcontext != NULL, "quantization library is not loaded");
    int ret = qcontext->dequant_func(buf, buf, count);
    ASSERT(ret == 0, "dequantization failed: error code %d", ret);
}

void quant_finalize()
{
    if (qcontext->quant_lib != NULL)
    {
        dlclose(qcontext->quant_lib);
        qcontext->quant_lib = NULL;
    }
    quant_free_diffs();
}
