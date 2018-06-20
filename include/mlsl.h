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
#ifndef MLSL_H
#define MLSL_H

#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C binding API version */
#define CMLSL_MAJOR_VERSION 1
#define CMLSL_MINOR_VERSION 0

#define CMLSL_VERSION(major, minor) ((major << 16) | (minor))
#define CMLSL_MAJOR(version)        (version >> 16)
#define CMLSL_MINOR(version)        (version & 0xFFFF)

#define CMLSL_VERSION_GE(v1, v2)    ((CMLSL_MAJOR(v1) > CMLSL_MAJOR(v2)) ||                                       \
                                    (CMLSL_MAJOR(v1) == CMLSL_MAJOR(v2) && CMLSL_MINOR(v1) == CMLSL_MINOR(v2)) || \
                                    (CMLSL_MAJOR(v1) == CMLSL_MAJOR(v2) && CMLSL_MINOR(v1) > CMLSL_MINOR(v2)))

#define CMLSL_VERSION_LT(v1, v2)    ((CMLSL_MAJOR(v1) < CMLSL_MAJOR(v2)) ||                                   \
                                    (CMLSL_MAJOR(v1) == CMLSL_MAJOR(v2) && CMLSL_MINOR(v1) < CMLSL_MINOR(v2)))

#define CMLSL_SUCCESS  0
#define CMLSL_FAILURE -1

#define PTR_TYPE unsigned long long
typedef PTR_TYPE mlsl_environment;
typedef PTR_TYPE mlsl_session;
typedef PTR_TYPE mlsl_operation;
typedef PTR_TYPE mlsl_operation_reg_info;
typedef PTR_TYPE mlsl_distribution;
typedef PTR_TYPE mlsl_parameter_set;
typedef PTR_TYPE mlsl_activation;
typedef PTR_TYPE mlsl_comm_block_info;
typedef PTR_TYPE mlsl_comm_req;
typedef PTR_TYPE mlsl_statistics;
typedef PTR_TYPE mlsl_quant_params;
typedef enum
{
    DT_FLOAT  = 0,
    DT_DOUBLE = 1,
    DT_BYTE   = 2
} mlsl_data_type;

typedef enum
{
    PT_TRAIN = 0,
    PT_TEST  = 1
} mlsl_phase_type;

typedef enum
{
    GT_DATA   = 0,
    GT_MODEL  = 1,
    GT_GLOBAL = 2
} mlsl_group_type;

typedef enum
{
    RT_SUM = 0,
    RT_MIN = 1,
    RT_MAX = 2
} mlsl_reduction_type;

typedef enum
{
    OT_CC     = 0,
    OT_BIAS   = 1,
    OT_ACT    = 2,
    OT_POOL   = 3,
    OT_SPLIT  = 4,
    OT_CONCAT = 5,
    OT_BCAST  = 6,
    OT_REDUCE = 7,
    OT_DATA   = 8,
    OT_EVAL   = 9
} mlsl_op_type;

typedef enum
{
    CT_NONE         = 0,
    CT_QUANTIZATION = 1
} mlsl_compression_type;
/*
typedef struct
{
    char* lib_path;
    char* quant_buffer_func_name;
    char* dequant_buffer_func_name;
    char* reduce_sum_func_name;
    size_t block_size;
    size_t elem_in_block;
} mlsl_quant_params;*/

/** mlsl_comm_block_info */
int mlsl_comm_block_info_get_mb_offset(mlsl_comm_block_info block_info, size_t* mb_offset);
int mlsl_comm_block_info_get_mb_count(mlsl_comm_block_info block_info, size_t* mb_count);
int mlsl_comm_block_info_get_fm_offset(mlsl_comm_block_info block_info, size_t* fm_offset);
int mlsl_comm_block_info_get_fm_count(mlsl_comm_block_info block_info, size_t* fm_count);
int mlsl_comm_block_info_get_fm_size(mlsl_comm_block_info block_info, size_t* fm_size);
int mlsl_comm_block_info_get_data_type(mlsl_comm_block_info block_info, mlsl_data_type* dtype);
int mlsl_comm_block_info_get_buf_offset(mlsl_comm_block_info block_info, size_t* buf_offset);

/** mlsl_activation */
int mlsl_activation_get_global_fm_count(mlsl_activation act, size_t* global_fm_count);
int mlsl_activation_get_global_fm_offset(mlsl_activation act, size_t* global_fm_offset);
int mlsl_activation_get_local_fm_count(mlsl_activation act, size_t* local_fm_count);
int mlsl_activation_get_pack_block_count(mlsl_activation act, size_t* pack_block_count);
int mlsl_activation_get_unpack_block_count(mlsl_activation act, size_t* unpack_block_count);
int mlsl_activation_get_pack_block(mlsl_activation act, size_t block_idx, mlsl_comm_block_info* block_info);
int mlsl_activation_get_unpack_block(mlsl_activation act, size_t block_idx, mlsl_comm_block_info* block_info);
int mlsl_activation_get_data_type(mlsl_activation act, mlsl_data_type* dtype);
int mlsl_activation_get_fm_size(mlsl_activation act, size_t* fm_size);
int mlsl_activation_get_comm_buf(mlsl_activation act, void** comm_buf);
int mlsl_activation_get_comm_buf_size(mlsl_activation act, size_t* size);
int mlsl_activation_start_comm(mlsl_activation act, void* buffer);
int mlsl_activation_wait_comm(mlsl_activation act, void** ret_buffer);

/** mlsl_parameter_set */
int mlsl_parameter_set_get_global_kernel_count(mlsl_parameter_set param_set, size_t* global_kernel_count);
int mlsl_parameter_set_get_global_kernel_offset(mlsl_parameter_set param_set, size_t* global_kernel_offset);
int mlsl_parameter_set_get_local_kernel_count(mlsl_parameter_set param_set, size_t* local_kernel_count);
int mlsl_parameter_set_get_owned_kernel_count(mlsl_parameter_set param_set, size_t* owned_kernel_count);
int mlsl_parameter_set_get_owned_kernel_offset(mlsl_parameter_set param_set, size_t* owned_kernel_offset);
int mlsl_parameter_set_get_data_type(mlsl_parameter_set param_set, mlsl_data_type* dtype);
int mlsl_parameter_set_get_kernel_size(mlsl_parameter_set param_set, size_t* kernel_size);
int mlsl_parameter_set_is_distributed_update(mlsl_parameter_set param_set, int* is_dist_update);
int mlsl_parameter_set_start_gradient_comm(mlsl_parameter_set param_set, void* buffer);
int mlsl_parameter_set_start_increment_comm(mlsl_parameter_set param_set, void* buffer);
int mlsl_parameter_set_wait_gradient_comm(mlsl_parameter_set param_set, void** ret_buffer);
int mlsl_parameter_set_test_gradient_comm(mlsl_parameter_set param_set, int* is_completed, void** ret_buffer);
int mlsl_parameter_set_wait_increment_comm(mlsl_parameter_set param_set, void** ret_buffer);

/** mlsl_distribution */
int mlsl_distribution_get_process_count(mlsl_distribution dist, mlsl_group_type group_type, size_t* process_count);
int mlsl_distribution_get_process_idx(mlsl_distribution dist, mlsl_group_type group_type, size_t* process_idx);
int mlsl_distribution_bcast(mlsl_distribution dist, void* buffer, size_t count, mlsl_data_type dtype,
                            size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_reduce(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t count, mlsl_data_type dtype,
                             mlsl_reduction_type red_type, size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_all_reduce(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t count, mlsl_data_type dtype,
                                 mlsl_reduction_type red_type, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_all_to_all(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                                 mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_all_to_allv(mlsl_distribution dist, void* send_buffer, size_t* send_counts, size_t* send_offsets, void* recv_buffer, size_t* recv_counts,
                                     size_t* recv_offsets, mlsl_data_type dtype, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_gather(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                             size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_all_gather(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                                 mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_scatter(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t recv_count, mlsl_data_type dtype,
                              size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_reduce_scatter(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t recv_count, mlsl_data_type dtype,
                                     mlsl_reduction_type red_type, mlsl_group_type group_type, mlsl_comm_req* req);
int mlsl_distribution_barrier(mlsl_distribution dist, mlsl_group_type group_type);

/** mlsl_operation_reg_info */
int mlsl_operation_reg_info_set_name(mlsl_operation_reg_info reg_info, const char* name);
int mlsl_operation_reg_info_add_input(mlsl_operation_reg_info reg_info, size_t fm_count, size_t fm_size, mlsl_data_type dtype);
int mlsl_operation_reg_info_add_output(mlsl_operation_reg_info reg_info, size_t fm_count, size_t fm_size, mlsl_data_type dtype);
int mlsl_operation_reg_info_add_parameter_set(mlsl_operation_reg_info reg_info, size_t kernel_count, size_t kernel_size,
                                              mlsl_data_type dtype, int dist_update);
int mlsl_operation_reg_info_add_parameter_set_with_compress(mlsl_operation_reg_info reg_info, size_t kernel_count, size_t kernel_size,
                                              mlsl_data_type dtype, int dist_update, mlsl_compression_type compress_type);
int mlsl_operation_reg_info_validate(mlsl_operation_reg_info reg_info, mlsl_distribution dist);

/** mlsl_operation */
int mlsl_operation_set_distribution(mlsl_operation op, mlsl_distribution dist);
int mlsl_operation_get_distribution(mlsl_operation op, mlsl_distribution* dist);
int mlsl_operation_get_session(mlsl_operation op, mlsl_session* session);
int mlsl_operation_get_op_type(mlsl_operation op, mlsl_op_type* op_type);
int mlsl_operation_set_prev(mlsl_operation op, mlsl_operation prev, size_t act_idx, size_t prev_op_act_idx);
int mlsl_operation_set_next(mlsl_operation op, mlsl_operation next, size_t act_idx, size_t next_op_act_idx);
int mlsl_operation_get_name(mlsl_operation op, const char** name);
int mlsl_operation_get_global_minibatch_size(mlsl_operation op, size_t* global_minibatch_size);
int mlsl_operation_get_local_minibatch_size(mlsl_operation op, size_t* local_minibatch_size);
int mlsl_operation_get_global_minibatch_offset(mlsl_operation op, size_t* global_minibatch_offset);
int mlsl_operation_get_input_count(mlsl_operation op, size_t* input_count);
int mlsl_operation_get_input(mlsl_operation op, size_t input_idx, mlsl_activation* input_act);
int mlsl_operation_get_output_count(mlsl_operation op, size_t* output_count);
int mlsl_operation_get_output(mlsl_operation op, size_t output_idx, mlsl_activation* output_act);
int mlsl_operation_has_parameter_sets(mlsl_operation op, int* has_params);
int mlsl_operation_get_parameter_set_count(mlsl_operation op, size_t* param_count);
int mlsl_operation_get_parameter_set(mlsl_operation op, size_t param_idx, mlsl_parameter_set* param_set);

/** mlsl_statistics */
int mlsl_statistics_start(mlsl_statistics stat);
int mlsl_statistics_stop(mlsl_statistics stat);
int mlsl_statistics_reset(mlsl_statistics stat);
int mlsl_statistics_print(mlsl_statistics stat);
int mlsl_statistics_is_started(mlsl_statistics stat, int* is_started);
int mlsl_statistics_is_enabled(mlsl_statistics stat, int* is_enabled);
int mlsl_statistics_get_isolation_comm_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles);
int mlsl_statistics_get_comm_size(mlsl_statistics stat, size_t op_idx, size_t* size);
int mlsl_statistics_get_comm_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles);
int mlsl_statistics_get_compute_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles);
int mlsl_statistics_get_total_isolation_comm_cycles(mlsl_statistics stat, unsigned long long* cycles);
int mlsl_statistics_get_total_comm_size(mlsl_statistics stat, size_t* size);
int mlsl_statistics_get_total_comm_cycles(mlsl_statistics stat, unsigned long long* cycles);
int mlsl_statistics_get_total_compute_cycles(mlsl_statistics stat, unsigned long long* cycles);

/** mlsl_session */
int mlsl_session_set_global_minibatch_size(mlsl_session session, size_t global_minibatch_size);
int mlsl_session_get_global_minibatch_size(mlsl_session session, size_t* global_minibatch_size);
int mlsl_session_get_phase_type(mlsl_session session, mlsl_phase_type* phase_type);
int mlsl_session_create_operation_reg_info(mlsl_session session, mlsl_op_type op_type, mlsl_operation_reg_info* reg_info);
int mlsl_session_delete_operation_reg_info(mlsl_session session, mlsl_operation_reg_info reg_info);
int mlsl_session_add_operation_with_distribution(mlsl_session session, mlsl_operation_reg_info reg_info,
                                                 mlsl_distribution dist, size_t* op_idx);
int mlsl_session_add_operation(mlsl_session session, mlsl_operation_reg_info reg_info, size_t* op_idx);
int mlsl_session_remove_operations(mlsl_session session);
int mlsl_session_get_operation_count(mlsl_session session, size_t* op_count);
int mlsl_session_get_operation(mlsl_session session, size_t op_idx, mlsl_operation* op);
int mlsl_session_commit(mlsl_session session);
int mlsl_session_get_stats(mlsl_session session, mlsl_statistics* stat);

/** mlsl_environment */
int mlsl_environment_get_env(mlsl_environment* env);
int mlsl_environment_get_version(int* version);
int mlsl_environment_configure(mlsl_environment env, const char* config);
int mlsl_environment_init(mlsl_environment env, int* argc, char** argv[]);
int mlsl_environment_finalize(mlsl_environment env);
int mlsl_environment_is_initialized(mlsl_environment env, int* is_initialized);
int mlsl_environment_get_process_idx(mlsl_environment env, size_t* process_idx);
int mlsl_environment_get_process_count(mlsl_environment env, size_t* process_count);
int mlsl_environment_create_session(mlsl_environment env, mlsl_phase_type phase_type, mlsl_session* session);
int mlsl_environment_delete_session(mlsl_environment env, mlsl_session session);
int mlsl_environment_create_distribution(mlsl_environment env, size_t data_partitions, size_t model_partitions, mlsl_distribution* dist);
int mlsl_environment_delete_distribution(mlsl_environment env, mlsl_distribution dist);
int mlsl_environment_wait(mlsl_environment env, mlsl_comm_req req);
int mlsl_environment_test(mlsl_environment env, mlsl_comm_req req, int* is_completed);
int mlsl_environment_alloc(mlsl_environment env, size_t size, size_t alignment, void** ptr);
int mlsl_environment_free(mlsl_environment env, void* ptr);
int mlsl_environment_set_quantization_params(mlsl_environment env, mlsl_quant_params* params);
int mlsl_environment_get_quantization_params(mlsl_environment env, mlsl_quant_params* params);
#ifdef __cplusplus
}
#endif

#endif /* MLSL_H */
