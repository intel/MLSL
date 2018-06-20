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

/* MLSL library C binding API usage example and correctness check test */

#include <math.h>  /* fabs */
#include <stdio.h> /* printf */

#include "mlsl.h"

/* Memory stuff */

#if defined(__INTEL_COMPILER) || defined(__ICC)
#define MY_MALLOC(size, align) _mm_malloc(size, align)
#define MY_FREE(ptr)           _mm_free(ptr)
#elif defined(__GNUC__)
#define MY_MALLOC(size, align) malloc(size)
#define MY_FREE(ptr)           free(ptr)
#else 
# error "this compiler is not supported" 
#endif


/* Logging stuff */

#define PRINT_BUF_LEN             4096
#define PRINT_BUF_FLUSH_THRESHOLD 3000
char print_buf[PRINT_BUF_LEN];
int print_count = 0;

#define MY_FLUSH()                \
  do                              \
  {                               \
      print_buf[print_count] = 0; \
      printf("%s", print_buf);    \
      print_count = 0;            \
      fflush(stdout);             \
  } while(0)

#define MY_PRINTF(...)                                                                     \
  do                                                                                       \
  {                                                                                        \
      int c = snprintf(print_buf + print_count, PRINT_BUF_LEN - print_count, __VA_ARGS__); \
      if (c > 0 && c < PRINT_BUF_LEN - print_count)                                        \
          print_count += c;                                                                \
      if (print_count > PRINT_BUF_FLUSH_THRESHOLD)                                         \
          MY_FLUSH();                                                                      \
  } while(0)

#define MY_ASSERT(cond,...)                                                   \
  do                                                                          \
  {                                                                           \
      if (!(cond))                                                            \
      {                                                                       \
          printf("%s:%d:assertion '%s' failed\n", __FILE__, __LINE__, #cond); \
          printf(__VA_ARGS__);                                                \
          mlsl_environment_finalize(env);                                     \
          exit(1);                                                            \
      }                                                                       \
  } while(0)


/* MLSL Test stuff */

#define MLSL_CALL(expression)                                             \
  do                                                                      \
  {                                                                       \
      int ret = expression;                                               \
      if (ret != CMLSL_SUCCESS)                                           \
      {                                                                   \
          printf("%s:%d: MLSL error: ret %d\n", __FILE__, __LINE__, ret); \
          mlsl_environment_finalize(env);                                 \
          exit(1);                                                        \
      }                                                                   \
  } while (0)

#define DTYPE                 float
#define DTYPE_SIZE            sizeof(DTYPE)
#define MLSL_DTYPE            (DTYPE_SIZE == 4) ? DT_FLOAT : DT_DOUBLE
#define CACHELINE_SIZE        64
#define FAIL_COUNTER_MAX      5

#define GLOBAL_MINIBATCH_SIZE 16
#define LAYER_COUNT           2
#define EPOCH_COUNT           4
#define MINIBATCH_PER_EPOCH   4

mlsl_environment env;
mlsl_session session;
mlsl_distribution distribution;

size_t process_idx;
size_t process_count;

/* default parameters */
size_t group_count = 1;
int use_dist_update = 1; // data parallelism's feature
int use_user_buf = 1;

typedef enum
{
    CONV_MIMO = 0,
    CONV_FLAT = 1,
    FC        = 2
} layer_type_t;

typedef struct
{
    size_t layer_idx;
    layer_type_t type;
    size_t ifm;
    size_t ofm;
    size_t ifm_width;
    size_t ifm_height;
    size_t ofm_width;
    size_t ofm_height;
    size_t kw;
    size_t kh;
} layer_params_t;
layer_params_t layer_params[LAYER_COUNT];

typedef struct
{
    mlsl_distribution distribution;
    mlsl_operation op;

    int has_param_set;
    mlsl_parameter_set param_set;
    size_t kernel_size;
    size_t local_kernel_count;
    size_t owned_kernel_count;
    size_t owned_kernel_offset;

    mlsl_activation input_act;
    size_t input_fm_size;
    size_t input_local_fm_count;
    size_t input_global_fm_offset;

    mlsl_activation output_act;
    size_t output_fm_size;
    size_t output_local_fm_count;
    size_t output_global_fm_offset;

    size_t local_mb_size;

    size_t data_group_size;
    size_t model_group_size;

    size_t layer_idx;
    DTYPE* input_act_buf, *output_act_buf;              /* input/output activations */
    DTYPE* input_act_grad_buf, *output_act_grad_buf;    /* gradients wrt input activations/ouput activations */
    DTYPE* param_buf, *param_grad_buf, *param_inc_buf;  /* learnable parameters, gradient wrt parameters, parameters increment */
    int is_backward_unpack_called;
} layer_t;
layer_t* layers[LAYER_COUNT];

void init_layer(layer_t* layer, size_t layer_idx, mlsl_operation op, layer_t* prev_layer)
{
    layer->op = op;
    layer->layer_idx = layer_idx;
    layer->is_backward_unpack_called = 0;

    MLSL_CALL(mlsl_operation_get_distribution(layer->op, &layer->distribution));

    MLSL_CALL(mlsl_operation_has_parameter_sets(layer->op, &layer->has_param_set));
    if (layer->has_param_set)
    {
        MLSL_CALL(mlsl_operation_get_parameter_set(layer->op, 0 /* idx */, &layer->param_set));
        MLSL_CALL(mlsl_parameter_set_get_kernel_size(layer->param_set, &layer->kernel_size));
        MLSL_CALL(mlsl_parameter_set_get_local_kernel_count(layer->param_set, &layer->local_kernel_count));
        MLSL_CALL(mlsl_parameter_set_get_owned_kernel_count(layer->param_set, &layer->owned_kernel_count));
        MLSL_CALL(mlsl_parameter_set_get_owned_kernel_offset(layer->param_set, &layer->owned_kernel_offset));
    }

    MLSL_CALL(mlsl_operation_get_input(layer->op, 0 /* idx */, &layer->input_act));
    MLSL_CALL(mlsl_activation_get_fm_size(layer->input_act, &layer->input_fm_size));
    MLSL_CALL(mlsl_activation_get_local_fm_count(layer->input_act, &layer->input_local_fm_count));
    MLSL_CALL(mlsl_activation_get_global_fm_offset(layer->input_act, &layer->input_global_fm_offset));

    MLSL_CALL(mlsl_operation_get_output(layer->op, 0 /* idx */, &layer->output_act));
    MLSL_CALL(mlsl_activation_get_fm_size(layer->output_act, &layer->output_fm_size));
    MLSL_CALL(mlsl_activation_get_local_fm_count(layer->output_act, &layer->output_local_fm_count));
    MLSL_CALL(mlsl_activation_get_global_fm_offset(layer->output_act, &layer->output_global_fm_offset));

    MLSL_CALL(mlsl_operation_get_local_minibatch_size(layer->op, &layer->local_mb_size));

    MLSL_CALL(mlsl_distribution_get_process_count(layer->distribution, GT_DATA, &layer->data_group_size));
    MLSL_CALL(mlsl_distribution_get_process_count(layer->distribution, GT_MODEL, &layer->model_group_size));

    size_t input_act_size, prev_output_act_size;

    if (prev_layer == NULL)
        prev_output_act_size = 0;
    else
    {
        size_t prev_local_fm_count;
        size_t prev_local_mb_size;
        size_t prev_fm_size;
        MLSL_CALL(mlsl_activation_get_local_fm_count(prev_layer->output_act, &prev_local_fm_count));
        MLSL_CALL(mlsl_operation_get_local_minibatch_size(prev_layer->op, &prev_local_mb_size));
        MLSL_CALL(mlsl_activation_get_fm_size(prev_layer->output_act, &prev_fm_size));
        prev_output_act_size = prev_local_fm_count * prev_local_mb_size * prev_fm_size * DTYPE_SIZE;
    }

    input_act_size = layer->input_local_fm_count * layer->local_mb_size * layer->input_fm_size * DTYPE_SIZE;
    if (prev_output_act_size > input_act_size) input_act_size = prev_output_act_size;

    layer->input_act_buf = (DTYPE*)MY_MALLOC(input_act_size, CACHELINE_SIZE);
    layer->input_act_grad_buf = (DTYPE*)MY_MALLOC(input_act_size, CACHELINE_SIZE);

    if (prev_layer != NULL)
    {
        prev_layer->output_act_buf = layer->input_act_buf;
        prev_layer->output_act_grad_buf = layer->input_act_grad_buf;
        MLSL_CALL(mlsl_operation_set_prev(layer->op, prev_layer->op, 0, 0));
    }

    size_t param_size = layer->local_kernel_count * layer->kernel_size * DTYPE_SIZE;
    size_t param_inc_size = layer->owned_kernel_count * layer->kernel_size * DTYPE_SIZE;

    if (use_user_buf)
    {
        layer->param_buf = (DTYPE*)MY_MALLOC(param_size, CACHELINE_SIZE);
        layer->param_grad_buf = (DTYPE*)MY_MALLOC(param_size, CACHELINE_SIZE);
        layer->param_inc_buf = (DTYPE*)MY_MALLOC(param_inc_size, CACHELINE_SIZE);
    }
    else
    {
        MLSL_CALL(mlsl_environment_alloc(env, param_size, CACHELINE_SIZE, (void**)&layer->param_buf));
        MLSL_CALL(mlsl_environment_alloc(env, param_size, CACHELINE_SIZE, (void**)&layer->param_grad_buf));
        MLSL_CALL(mlsl_environment_alloc(env, param_inc_size, CACHELINE_SIZE, (void**)&layer->param_inc_buf));
    }

    MY_ASSERT(layer->input_act_buf && layer->input_act_grad_buf && layer->param_buf && layer->param_grad_buf && layer->param_inc_buf,
              "error while buffers allocating");

    size_t idx = 0;
    for (; idx < param_size / DTYPE_SIZE; idx++)
        layer->param_buf[idx] = idx;
}

void finalize_layer(layer_t* layer)
{
    MY_FREE(layer->input_act_buf);
    MY_FREE(layer->input_act_grad_buf);

    if (use_user_buf)
    {
        MY_FREE(layer->param_buf);
        MY_FREE(layer->param_grad_buf);
        MY_FREE(layer->param_inc_buf);
    }
    else
    {
        MLSL_CALL(mlsl_environment_free(env, layer->param_buf));
        MLSL_CALL(mlsl_environment_free(env, layer->param_grad_buf));
        MLSL_CALL(mlsl_environment_free(env, layer->param_inc_buf));
    }
}

void pack_buffer(mlsl_activation act, DTYPE* comm_buf, DTYPE* local_buf)
{
    size_t local_fm_count, block_count;
    MLSL_CALL(mlsl_activation_get_local_fm_count(act, &local_fm_count));
    MLSL_CALL(mlsl_activation_get_pack_block_count(act, &block_count));

    size_t block_idx = 0;
    for (; block_idx < block_count; block_idx++)
    {
        mlsl_comm_block_info block;
        MLSL_CALL(mlsl_activation_get_pack_block(act, block_idx, &block));

        size_t mb_count, mb_offset, fm_count, fm_offset, fm_size, buf_offset;
        MLSL_CALL(mlsl_comm_block_info_get_mb_count(block, &mb_count));
        MLSL_CALL(mlsl_comm_block_info_get_mb_offset(block, &mb_offset));
        MLSL_CALL(mlsl_comm_block_info_get_fm_count(block, &fm_count));
        MLSL_CALL(mlsl_comm_block_info_get_fm_offset(block, &fm_offset));
        MLSL_CALL(mlsl_comm_block_info_get_fm_size(block, &fm_size));
        MLSL_CALL(mlsl_comm_block_info_get_buf_offset(block, &buf_offset));

        DTYPE* src = local_buf;
        DTYPE* dst = comm_buf + buf_offset;

        size_t mb_idx = 0, fm_idx = 0, space_idx = 0;
        for (mb_idx = 0; mb_idx < mb_count; mb_idx++)
            for (fm_idx = 0; fm_idx < fm_count; fm_idx++)
                for (space_idx = 0 ; space_idx < fm_size; space_idx++)
                    dst[mb_idx * fm_count * fm_size + fm_idx * fm_size + space_idx]
                        = src[(mb_idx + mb_offset) * local_fm_count * fm_size + (fm_idx + fm_offset) * fm_size + space_idx];
    }
}

void unpack_buffer(mlsl_activation act, DTYPE* comm_buf, DTYPE* local_buf)
{
    size_t local_fm_count, block_count;
    MLSL_CALL(mlsl_activation_get_local_fm_count(act, &local_fm_count));
    MLSL_CALL(mlsl_activation_get_unpack_block_count(act, &block_count));

    size_t block_idx = 0;
    for (; block_idx < block_count; block_idx++)
    {
        mlsl_comm_block_info block;
        MLSL_CALL(mlsl_activation_get_unpack_block(act, block_idx, &block));

        size_t mb_count, mb_offset, fm_count, fm_offset, fm_size, buf_offset;
        MLSL_CALL(mlsl_comm_block_info_get_mb_count(block, &mb_count));
        MLSL_CALL(mlsl_comm_block_info_get_mb_offset(block, &mb_offset));
        MLSL_CALL(mlsl_comm_block_info_get_fm_count(block, &fm_count));
        MLSL_CALL(mlsl_comm_block_info_get_fm_offset(block, &fm_offset));
        MLSL_CALL(mlsl_comm_block_info_get_fm_size(block, &fm_size));
        MLSL_CALL(mlsl_comm_block_info_get_buf_offset(block, &buf_offset));

        DTYPE* src = comm_buf + buf_offset;
        DTYPE* dst = local_buf;

        size_t mb_idx = 0, fm_idx = 0, space_idx = 0;
        for (mb_idx = 0; mb_idx < mb_count; mb_idx++)
            for (fm_idx = 0; fm_idx < fm_count; fm_idx++)
                for (space_idx = 0 ; space_idx < fm_size; space_idx++)
                    dst[(mb_idx + mb_offset) * local_fm_count * fm_size + (fm_idx + fm_offset) * fm_size + space_idx]
                        = src[mb_idx * fm_count * fm_size + fm_idx * fm_size + space_idx];
    }
}

void forward_compute(layer_t* layer, DTYPE* input_act, DTYPE* param, DTYPE* output_act)
{
    if (layer->layer_idx == 0)
    {
        /* Write to output activation */
        size_t out_size = layer->output_local_fm_count * layer->local_mb_size * layer->output_fm_size;
        size_t idx = 0;
        for (; idx < out_size; idx++)
            output_act[idx] = idx;
    }
    else if (layer->layer_idx == 1)
    {
        /* Check for input activation */
        size_t fm_local_count = layer->input_local_fm_count;
        size_t mb_local_len = layer->local_mb_size;
        size_t fm_size = layer->input_fm_size;
        size_t fm_offset =  layer->input_global_fm_offset;
        size_t fm_group_size = layer->model_group_size;
        size_t fail_counter = 0;

        size_t mb_idx = 0, fm_idx = 0, space_idx = 0;
        for (mb_idx = 0; mb_idx < mb_local_len; mb_idx++)
        {
            for (fm_idx = 0; fm_idx < fm_local_count; fm_idx++)
            {
                for (space_idx = 0; space_idx < fm_size; space_idx++)
                {
                    DTYPE expected = fm_group_size * (mb_idx * fm_local_count * fm_size * fm_group_size + (fm_offset + fm_idx) * fm_size + space_idx);
                    size_t idx = mb_idx * fm_local_count * fm_size + fm_idx * fm_size + space_idx;
                    if (fabs(input_act[idx] - expected) > 1.e-4)
                    {
                        if (fail_counter < FAIL_COUNTER_MAX)
                            MY_PRINTF("[%zu] forward input: idx %zu: expected %4.0f - received: %4.0f\n", process_idx, idx, expected, input_act[idx]);
                        fail_counter++;
                    }
                }
            }
        }

        if (fail_counter > 0)
            MY_PRINTF("[%zu] forward input activation test FAILED mismatch count = %zu\n", process_idx, fail_counter);
        else
            MY_PRINTF("[%zu] forward input activation test PASSED\n", process_idx);
    }

    /* Now check ParameterSet */
    size_t param_size = layer->local_kernel_count * layer->kernel_size;
    size_t fail_counter = 0;
    size_t idx = 0;
    for (; idx < param_size; idx++)
    {
        if (fabs(param[idx] - idx) > 1.e-4)
        {
            if (fail_counter < FAIL_COUNTER_MAX)
                MY_PRINTF("[%zu] forward parameter: idx %zu: expected %4.0f - received: %4.0f\n",
                          process_idx,
                          idx,
                          (DTYPE)idx,
                          param[idx]);
            fail_counter++;
        }
    }

    if (fail_counter > 0)
        MY_PRINTF("[%zu] forward parameter test FAILED mismatch count = %zu\n", process_idx, fail_counter);
    else
        MY_PRINTF("[%zu] forward parameter test PASSED\n", process_idx);
    MY_FLUSH();
}

void backward_compute1(layer_t* layer, DTYPE* output_act_grad, DTYPE* param, DTYPE* input_act_grad)
{
    if (layer->layer_idx == 0)
    {
        /* Check for inputs */
        size_t act_size = layer->output_local_fm_count * layer->local_mb_size * layer->output_fm_size;
        size_t fail_counter = 0;
        size_t idx = 0;
        for (; idx < act_size; idx++)
        {
            if (fabs(output_act_grad[idx] - idx) > 1.e-4)
            {
                if (fail_counter < FAIL_COUNTER_MAX)
                    MY_PRINTF("[%zu] backward output activation gradient: idx %zu: expected %4.0f - received: %4.0f\n",
                             process_idx,
                             idx,
                             (DTYPE)idx,
                             output_act_grad[idx]);
                fail_counter++;
            }
        }
        if (fail_counter > 0)
            MY_PRINTF("[%zu] backward output activation gradient test FAILED mismatch count = %zu\n", process_idx, fail_counter);
        else
            MY_PRINTF("[%zu] backward output activation gradient test PASSED\n", process_idx);
    }
    else if (layer->layer_idx == 1)
    {
        /* Write to output */
        size_t fm_local_count = layer->input_local_fm_count;
        size_t mb_local_len = layer->local_mb_size;
        size_t fm_size = layer->input_fm_size;
        size_t act_offset =  layer->input_global_fm_offset;
        size_t group_size = layer->model_group_size;

        size_t mb_idx = 0, fm_idx = 0, space_idx = 0;
        for (mb_idx = 0; mb_idx < mb_local_len; mb_idx++)
            for (fm_idx = 0; fm_idx < fm_local_count; fm_idx++)
                for (space_idx = 0; space_idx < fm_size; space_idx++)
                {
                    size_t idx = mb_idx * fm_local_count * fm_size + fm_idx * fm_size + space_idx;
                    input_act_grad[idx] = mb_idx * fm_local_count * fm_size * group_size + (act_offset + fm_idx) * fm_size + space_idx;
                }
    }
    MY_FLUSH();
}

void backward_compute2(layer_t* layer, DTYPE* output_act_grad, DTYPE* input_act, DTYPE* param_grad)
{
    size_t param_size = layer->local_kernel_count * layer->kernel_size;
    size_t idx = 0;
    for (; idx < param_size; idx++)
        param_grad[idx] = idx;
}

void update_compute(layer_t* layer, DTYPE* param_grad, DTYPE* param_inc, DTYPE* owned_param, size_t owned_size)
{
    size_t mb_group_size = layer->data_group_size;
    size_t owned_offset = layer->owned_kernel_offset * layer->kernel_size;
    size_t fail_counter = 0;
    size_t idx = 0;
    for (; idx < owned_size; idx++)
    {
        DTYPE expected = mb_group_size * (owned_offset + idx);
        if (fabs(param_grad[idx] - expected) > 1.e-4)
            fail_counter++;
        owned_param[idx] = owned_offset + idx;
    }
    if (fail_counter > 0)
        MY_PRINTF("[%zu] parameter gradient test FAILED mismatch count = %zu\n", process_idx, fail_counter);
    else
        MY_PRINTF("[%zu] parameter gradient test PASSED\n", process_idx);
    MY_FLUSH();
}

/* Recv parameter increments (in case of distributed update) and input activations, send output activations */
void forward(layer_t* layer)
{
    DTYPE* input_act_comm_buf;
    MLSL_CALL(mlsl_activation_wait_comm(layer->input_act, (void**)&input_act_comm_buf));
    unpack_buffer(layer->input_act, input_act_comm_buf, layer->input_act_buf);
    if (layer->has_param_set)
    {
        DTYPE* inc_comm_buf;
        MLSL_CALL(mlsl_parameter_set_wait_increment_comm(layer->param_set, (void**)&inc_comm_buf));
    }

    forward_compute(layer, layer->input_act_buf, layer->param_buf, layer->output_act_buf);

    DTYPE* output_act_comm_buf;
    MLSL_CALL(mlsl_activation_get_comm_buf(layer->output_act, (void**)&output_act_comm_buf));
    pack_buffer(layer->output_act, output_act_comm_buf, layer->output_act_buf);
    MLSL_CALL(mlsl_activation_start_comm(layer->output_act, output_act_comm_buf));
    layer->is_backward_unpack_called = 0;
}

/* Calculate gradient wrt input activation and send it */
void backward1(layer_t* layer)
{
    if (!layer->is_backward_unpack_called)
    {
        DTYPE* output_act_comm_buf;
        MLSL_CALL(mlsl_activation_wait_comm(layer->output_act, (void**)&output_act_comm_buf));
        unpack_buffer(layer->output_act, output_act_comm_buf, layer->output_act_grad_buf);
        layer->is_backward_unpack_called = 1;
    }

    backward_compute1(layer, layer->output_act_grad_buf, layer->param_buf, layer->input_act_grad_buf);

    DTYPE* input_act_comm_buf;
    MLSL_CALL(mlsl_activation_get_comm_buf(layer->input_act, (void**)&input_act_comm_buf));
    pack_buffer(layer->input_act, input_act_comm_buf, layer->input_act_grad_buf);
    MLSL_CALL(mlsl_activation_start_comm(layer->input_act, input_act_comm_buf));
}

/* Calculate gradient wrt parameters and send it */
void backward2(layer_t* layer)
{
    if (!layer->is_backward_unpack_called)
    {
        DTYPE* comm_buf;
        MLSL_CALL(mlsl_activation_wait_comm(layer->output_act, (void**)&comm_buf));
        unpack_buffer(layer->output_act, comm_buf, layer->output_act_grad_buf);
        layer->is_backward_unpack_called = 1;
    }

    backward_compute2(layer, layer->output_act_grad_buf, layer->input_act_buf, layer->param_grad_buf);

    if (layer->has_param_set)
        MLSL_CALL(mlsl_parameter_set_start_gradient_comm(layer->param_set, layer->param_grad_buf));
}

/* Recv gradient wrt parameters and update parameters/send parameter increments (in case of distributed update) */
void update(layer_t* layer)
{
    if (layer->has_param_set)
    {
        DTYPE* comm_buf;
        MLSL_CALL(mlsl_parameter_set_wait_gradient_comm(layer->param_set, (void**)&comm_buf));
        DTYPE* owned_param_buf = layer->param_buf + layer->owned_kernel_offset * layer->kernel_size;

        update_compute(layer, comm_buf == NULL ? layer->param_grad_buf : comm_buf,
                      layer->param_inc_buf,
                      owned_param_buf,
                      layer->owned_kernel_count * layer->kernel_size);

        MLSL_CALL(mlsl_parameter_set_start_increment_comm(layer->param_set, layer->param_buf));
    }
}


/* Layer initialization */
layer_t* create_layer(mlsl_session ses, mlsl_distribution dist, layer_type_t type, layer_params_t* lparams,  layer_t* prev_layer)
{
    MY_ASSERT((type == CONV_MIMO || type == CONV_FLAT || type == FC), "incorrect op type");

    size_t layer_idx = lparams->layer_idx;

    mlsl_operation_reg_info reg_info;
    MLSL_CALL(mlsl_session_create_operation_reg_info(ses, OT_CC, &reg_info));
    MLSL_CALL(mlsl_operation_reg_info_set_name(reg_info, "MyLayerName"));
    MLSL_CALL(mlsl_operation_reg_info_add_input(reg_info, lparams->ifm, lparams->ifm_width * lparams->ifm_height, MLSL_DTYPE));
    MLSL_CALL(mlsl_operation_reg_info_add_output(reg_info, lparams->ofm, lparams->ofm_width * lparams->ofm_height, MLSL_DTYPE));
    MLSL_CALL(mlsl_operation_reg_info_add_parameter_set(reg_info, lparams->ifm * lparams->ofm, lparams->kw * lparams->kh,
                                                        MLSL_DTYPE, use_dist_update));

    size_t op_idx;
    MLSL_CALL(mlsl_session_add_operation_with_distribution(ses, reg_info, dist, &op_idx));
    MLSL_CALL(mlsl_session_delete_operation_reg_info(ses, reg_info));

    mlsl_operation op;
    MLSL_CALL(mlsl_session_get_operation(ses, op_idx, &op));
    layer_t* layer = (layer_t*)malloc(sizeof(layer_t));
    init_layer(layer, layer_idx, op, prev_layer);

    return layer;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("specify parameters: cmlsl_test GROUP_COUNT [DIST_UPDATE] [USER_BUF]\n");
        exit(0);
    }

    int runtime_version;
    MLSL_CALL(mlsl_environment_get_version(&runtime_version));
    printf("built with CMLSL API version: %d.%d, used CMLSL API version: %d.%d\n",
           CMLSL_MAJOR_VERSION, CMLSL_MINOR_VERSION, CMLSL_MAJOR(runtime_version), CMLSL_MINOR(runtime_version));

    if (CMLSL_MAJOR_VERSION != CMLSL_MAJOR(runtime_version))
    {
        printf("incompatible MLSL C API version: %d.%d, exit\n",
               CMLSL_MAJOR(runtime_version), CMLSL_MINOR(runtime_version));
        return 0;
    }

    MLSL_CALL(mlsl_environment_get_env(&env));
    MLSL_CALL(mlsl_environment_init(env, &argc, &argv));
    MLSL_CALL(mlsl_environment_create_session(env, PT_TRAIN, &session));
    MLSL_CALL(mlsl_session_set_global_minibatch_size(session, GLOBAL_MINIBATCH_SIZE));
    MLSL_CALL(mlsl_environment_get_process_count(env, &process_count));

    if (argc > 1) group_count     = atoi(argv[1]);
    if (argc > 2) use_dist_update = atoi(argv[2]);
    if (argc > 3) use_user_buf    = atoi(argv[3]);

    if (group_count < 1) group_count = 1;
    if (group_count > process_count) group_count = process_count;

    MLSL_CALL(mlsl_environment_get_process_idx(env, &process_idx));
    if (process_idx == 0)
        printf("\nprocess_count = %zu, distribution = %zu x %zu (data_parts x model_parts), dist_update %d, user_buf %d\n\n",
               process_count, process_count/group_count, group_count, use_dist_update, use_user_buf);

    /* Correctness test assumes both the layers use same distribution */
    MLSL_CALL(mlsl_environment_create_distribution(env, process_count/group_count, group_count, &distribution));

    /* Init all the layers */
    size_t layer_idx;
    for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
    {
        /* Set layer_params for each layer */
        if (layer_idx == 0)
        {
            layer_params[layer_idx].layer_idx = layer_idx;
            layer_params[layer_idx].type = CONV_MIMO;
            layer_params[layer_idx].ifm  = 128;
            layer_params[layer_idx].ofm  = 256;
            layer_params[layer_idx].ifm_width = 12;
            layer_params[layer_idx].ifm_height = 12;
            layer_params[layer_idx].ofm_width = 12;
            layer_params[layer_idx].ofm_height = 12;
            layer_params[layer_idx].kw = 3;
            layer_params[layer_idx].kh = 3;
        }
        else if (layer_idx == 1)
        {
            layer_params[layer_idx].layer_idx = layer_idx;
            layer_params[layer_idx].type = CONV_MIMO;
            layer_params[layer_idx].ifm  = 256;
            layer_params[layer_idx].ofm  = 256;
            layer_params[layer_idx].ifm_width = 12;
            layer_params[layer_idx].ifm_height = 12;
            layer_params[layer_idx].ofm_width = 12;
            layer_params[layer_idx].ofm_height = 12;
            layer_params[layer_idx].kw = 3;
            layer_params[layer_idx].kh = 3;
        }

        layers[layer_idx] = create_layer(session,
                                         distribution,
                                         layer_params[layer_idx].type,
                                         &layer_params[layer_idx],
                                         (layer_idx == 0 ? NULL : layers[layer_idx - 1]));
    }
    MLSL_CALL(mlsl_session_commit(session));

    size_t epoch_idx = 0, mb_idx = 0;
    for (epoch_idx = 0; epoch_idx < EPOCH_COUNT; epoch_idx++)
    {
        for (mb_idx = 0; mb_idx < MINIBATCH_PER_EPOCH; mb_idx++)
        {
            for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
                forward(layers[layer_idx]);

            for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
            {
                /* Split backward phase on 2 steps to achieve comp/comm overlapping */
                backward1(layers[LAYER_COUNT - layer_idx - 1]);
                backward2(layers[LAYER_COUNT - layer_idx - 1]);
            }

            for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
                update(layers[layer_idx]);
        }

        /* Finish ParameterSet comms before ending epoch */
        for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
        {
            if (layers[layer_idx]->has_param_set)
            {
                DTYPE* inc_comm_buf;
                MLSL_CALL(mlsl_parameter_set_wait_increment_comm(layers[layer_idx]->param_set, (void**)&inc_comm_buf));
            }
        }
    }

    for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx++)
    {
        finalize_layer(layers[layer_idx]);
        free(layers[layer_idx]);
    }

    MLSL_CALL(mlsl_environment_delete_session(env, session));
    MLSL_CALL(mlsl_environment_delete_distribution(env, distribution));
    MLSL_CALL(mlsl_environment_finalize(env));

    printf("[%zu] exited normally\n", process_idx);

    return 0;
}
