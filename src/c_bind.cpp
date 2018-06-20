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
#include "mlsl.h"
#include "mlsl.hpp"

#include <stdio.h>

using namespace MLSL;

/** HNDL (handle) is C-binding's object which stores pointer to C++ class object */

#define HNDL_TO_MLSL(handle, mlslType) (reinterpret_cast<mlslType>((void*)handle))
#define MLSL_TO_HNDL(object, handleType) (reinterpret_cast<handleType>(object))

#define TO_MLSL(object, mlslType) (static_cast<mlslType>(object))
#define TO_CMLSL(object, cmlslType) (static_cast<cmlslType>(object))

#define TRY_CATCH_RETURN(expression) \
  try                                \
  {                                  \
      expression;                    \
  }                                  \
  catch (...)                        \
  {                                  \
      return CMLSL_FAILURE;          \
  }                                  \
  return CMLSL_SUCCESS;


/** mlsl_comm_block_info */

int mlsl_comm_block_info_get_mb_offset(mlsl_comm_block_info block_info, size_t* mb_offset)
{
    TRY_CATCH_RETURN((*mb_offset) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetMbOffset());
}

int mlsl_comm_block_info_get_mb_count(mlsl_comm_block_info block_info, size_t* mb_count)
{
    TRY_CATCH_RETURN((*mb_count) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetMbCount());
}

int mlsl_comm_block_info_get_fm_offset(mlsl_comm_block_info block_info, size_t* fm_offset)
{
    TRY_CATCH_RETURN((*fm_offset) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetFmOffset());
}

int mlsl_comm_block_info_get_fm_count(mlsl_comm_block_info block_info, size_t* fm_count)
{
    TRY_CATCH_RETURN((*fm_count) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetFmCount());
}

int mlsl_comm_block_info_get_fm_size(mlsl_comm_block_info block_info, size_t* fm_size)
{
    TRY_CATCH_RETURN((*fm_size) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetFmSize());
}

int mlsl_comm_block_info_get_data_type(mlsl_comm_block_info block_info, mlsl_data_type* dtype)
{
    TRY_CATCH_RETURN((*dtype) = TO_CMLSL(HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetDataType(), mlsl_data_type));
}

int mlsl_comm_block_info_get_buf_offset(mlsl_comm_block_info block_info, size_t* buf_offset)
{
    TRY_CATCH_RETURN((*buf_offset) = HNDL_TO_MLSL(block_info, CommBlockInfo*)->GetBufOffset());
}


/** mlsl_activation */

int mlsl_activation_get_global_fm_count(mlsl_activation act, size_t* global_fm_count)
{
    TRY_CATCH_RETURN((*global_fm_count) = HNDL_TO_MLSL(act, Activation*)->GetGlobalFmCount());
}

int mlsl_activation_get_global_fm_offset(mlsl_activation act, size_t* global_fm_offset)
{
    TRY_CATCH_RETURN((*global_fm_offset) = HNDL_TO_MLSL(act, Activation*)->GetGlobalFmOffset());
}

int mlsl_activation_get_local_fm_count(mlsl_activation act, size_t* local_fm_count)
{
    TRY_CATCH_RETURN((*local_fm_count) = HNDL_TO_MLSL(act, Activation*)->GetLocalFmCount());
}

int mlsl_activation_get_pack_block_count(mlsl_activation act, size_t* pack_block_count)
{
    TRY_CATCH_RETURN((*pack_block_count) = HNDL_TO_MLSL(act, Activation*)->GetPackBlockCount());
}

int mlsl_activation_get_unpack_block_count(mlsl_activation act, size_t* unpack_block_count)
{
    TRY_CATCH_RETURN((*unpack_block_count) = HNDL_TO_MLSL(act, Activation*)->GetUnpackBlockCount());
}

int mlsl_activation_get_pack_block(mlsl_activation act, size_t block_idx, mlsl_comm_block_info* block_info)
{
    TRY_CATCH_RETURN((*block_info) = MLSL_TO_HNDL(HNDL_TO_MLSL(act, Activation*)->GetPackBlock(block_idx), mlsl_comm_block_info));
}

int mlsl_activation_get_unpack_block(mlsl_activation act, size_t block_idx, mlsl_comm_block_info* block_info)
{
    TRY_CATCH_RETURN((*block_info) = MLSL_TO_HNDL(HNDL_TO_MLSL(act, Activation*)->GetUnpackBlock(block_idx), mlsl_comm_block_info));
}

int mlsl_activation_get_data_type(mlsl_activation act, mlsl_data_type* dtype)
{
    TRY_CATCH_RETURN((*dtype) = TO_CMLSL(HNDL_TO_MLSL(act, Activation*)->GetDataType(), mlsl_data_type));
}

int mlsl_activation_get_fm_size(mlsl_activation act, size_t* fm_size)
{
    TRY_CATCH_RETURN((*fm_size) = HNDL_TO_MLSL(act, Activation*)->GetFmSize());
}

int mlsl_activation_get_comm_buf(mlsl_activation act, void** comm_buf)
{
    TRY_CATCH_RETURN((*comm_buf) = HNDL_TO_MLSL(act, Activation*)->GetCommBuf());
}

int mlsl_activation_get_comm_buf_size(mlsl_activation act, size_t* size)
{
    TRY_CATCH_RETURN((*size) = HNDL_TO_MLSL(act, Activation*)->GetCommBufSize());
}

int mlsl_activation_start_comm(mlsl_activation act, void* buffer)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(act, Activation*)->StartComm(buffer));
}

int mlsl_activation_wait_comm(mlsl_activation act, void** ret_buffer)
{
    TRY_CATCH_RETURN((*ret_buffer) = HNDL_TO_MLSL(act, Activation*)->WaitComm());
}


/** mlsl_parameter_set */

int mlsl_parameter_set_get_global_kernel_count(mlsl_parameter_set param_set, size_t* global_kernel_count)
{
    TRY_CATCH_RETURN((*global_kernel_count) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetGlobalKernelCount());
}

int mlsl_parameter_set_get_global_kernel_offset(mlsl_parameter_set param_set, size_t* global_kernel_offset)
{
    TRY_CATCH_RETURN((*global_kernel_offset) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetGlobalKernelOffset());
}

int mlsl_parameter_set_get_local_kernel_count(mlsl_parameter_set param_set, size_t* local_kernel_count)
{
    TRY_CATCH_RETURN((*local_kernel_count) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetLocalKernelCount());
}

int mlsl_parameter_set_get_owned_kernel_count(mlsl_parameter_set param_set, size_t* owned_kernel_count)
{
    TRY_CATCH_RETURN((*owned_kernel_count) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetOwnedKernelCount());
}

int mlsl_parameter_set_get_owned_kernel_offset(mlsl_parameter_set param_set, size_t* owned_kernel_offset)
{
    TRY_CATCH_RETURN((*owned_kernel_offset) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetOwnedKernelOffset());
}

int mlsl_parameter_set_get_data_type(mlsl_parameter_set param_set, mlsl_data_type* dtype)
{
    TRY_CATCH_RETURN((*dtype) = TO_CMLSL(HNDL_TO_MLSL(param_set, ParameterSet*)->GetDataType(), mlsl_data_type));
}

int mlsl_parameter_set_get_kernel_size(mlsl_parameter_set param_set, size_t* kernel_size)
{
    TRY_CATCH_RETURN((*kernel_size) = HNDL_TO_MLSL(param_set, ParameterSet*)->GetKernelSize());
}

int mlsl_parameter_set_is_distributed_update(mlsl_parameter_set param_set, int* is_dist_update)
{
    TRY_CATCH_RETURN((*is_dist_update) = (HNDL_TO_MLSL(param_set, ParameterSet*)->IsDistributedUpdate() ? 1 : 0));
}

int mlsl_parameter_set_start_gradient_comm(mlsl_parameter_set param_set, void* buffer)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(param_set, ParameterSet*)->StartGradientComm(buffer));
}

int mlsl_parameter_set_start_increment_comm(mlsl_parameter_set param_set, void* buffer)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(param_set, ParameterSet*)->StartIncrementComm(buffer));
}

int mlsl_parameter_set_wait_gradient_comm(mlsl_parameter_set param_set, void** ret_buffer)
{
    TRY_CATCH_RETURN((*ret_buffer) = HNDL_TO_MLSL(param_set, ParameterSet*)->WaitGradientComm());
}

int mlsl_parameter_set_test_gradient_comm(mlsl_parameter_set param_set, int* is_completed, void** ret_buffer)
{
    TRY_CATCH_RETURN((*ret_buffer) = HNDL_TO_MLSL(param_set, ParameterSet*)->TestGradientComm((bool*)is_completed));
}

int mlsl_parameter_set_wait_increment_comm(mlsl_parameter_set param_set, void** ret_buffer)
{
    TRY_CATCH_RETURN((*ret_buffer) = HNDL_TO_MLSL(param_set, ParameterSet*)->WaitIncrementComm());
}


/** mlsl_distribution */

int mlsl_distribution_get_process_count(mlsl_distribution dist, mlsl_group_type group_type, size_t* process_count)
{
    TRY_CATCH_RETURN((*process_count) = HNDL_TO_MLSL(dist, Distribution*)->GetProcessCount(TO_MLSL(group_type, GroupType)));
}

int mlsl_distribution_get_process_idx(mlsl_distribution dist, mlsl_group_type group_type, size_t* process_idx)
{
    TRY_CATCH_RETURN((*process_idx) = HNDL_TO_MLSL(dist, Distribution*)->GetProcessIdx(TO_MLSL(group_type, GroupType)));
}

int mlsl_distribution_bcast(mlsl_distribution dist, void* buffer, size_t count, mlsl_data_type dtype,
                            size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->Bcast(buffer, count, TO_MLSL(dtype, DataType), root_idx,
                                                                                    TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_reduce(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t count, mlsl_data_type dtype,
                             mlsl_reduction_type red_type, size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->Reduce(send_buffer, recv_buffer, count, TO_MLSL(dtype, DataType),
                                                                                     TO_MLSL(red_type, ReductionType),
                                                                                     root_idx, TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_all_reduce(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t count, mlsl_data_type dtype,
                                 mlsl_reduction_type red_type, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->AllReduce(send_buffer, recv_buffer, count, TO_MLSL(dtype, DataType),
                                                                                        TO_MLSL(red_type, ReductionType),
                                                                                        TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}
int mlsl_distribution_all_to_all(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                                 mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->AlltoAll(send_buffer, send_count, recv_buffer, TO_MLSL(dtype, DataType),
                                                                                       TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_all_to_allv(mlsl_distribution dist, void* send_buffer, size_t* send_counts, size_t* send_offsets, void* recv_buffer, size_t* recv_counts,
                                  size_t* recv_offsets, mlsl_data_type dtype, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->AlltoAllv(send_buffer, send_counts, send_offsets, recv_buffer, recv_counts,
                                                                                        recv_offsets, TO_MLSL(dtype, DataType), TO_MLSL(group_type, GroupType)),
                                                                                        mlsl_comm_req));
}

int mlsl_distribution_gather(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                             size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->Gather(send_buffer, send_count, recv_buffer, TO_MLSL(dtype, DataType),
                                                                                     root_idx, TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_all_gather(mlsl_distribution dist, void* send_buffer, size_t send_count, void* recv_buffer, mlsl_data_type dtype,
                                 mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->AllGather(send_buffer, send_count, recv_buffer, TO_MLSL(dtype, DataType),
                                                                                        TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_scatter(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t recv_count, mlsl_data_type dtype,
                              size_t root_idx, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->Scatter(send_buffer, recv_buffer, recv_count, TO_MLSL(dtype, DataType),
                                                                                      root_idx, TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_reduce_scatter(mlsl_distribution dist, void* send_buffer, void* recv_buffer, size_t recv_count, mlsl_data_type dtype,
                                     mlsl_reduction_type red_type, mlsl_group_type group_type, mlsl_comm_req* req)
{
    TRY_CATCH_RETURN((*req) = MLSL_TO_HNDL(HNDL_TO_MLSL(dist, Distribution*)->ReduceScatter(send_buffer, recv_buffer, recv_count, TO_MLSL(dtype, DataType),
                                                                                            TO_MLSL(red_type, ReductionType),
                                                                                            TO_MLSL(group_type, GroupType)), mlsl_comm_req));
}

int mlsl_distribution_barrier(mlsl_distribution dist, mlsl_group_type group_type)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(dist, Distribution*)->Barrier(TO_MLSL(group_type, GroupType)));
}


/** mlsl_operation_reg_info */

int mlsl_operation_reg_info_set_name(mlsl_operation_reg_info reg_info, const char* name)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->SetName(name));
}

int mlsl_operation_reg_info_add_input(mlsl_operation_reg_info reg_info, size_t fm_count, size_t fm_size, mlsl_data_type dtype)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->AddInput(fm_count, fm_size, TO_MLSL(dtype, DataType)));
}

int mlsl_operation_reg_info_add_output(mlsl_operation_reg_info reg_info, size_t fm_count, size_t fm_size, mlsl_data_type dtype)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->AddOutput(fm_count, fm_size, TO_MLSL(dtype, DataType)));
}

int mlsl_operation_reg_info_add_parameter_set(mlsl_operation_reg_info reg_info, size_t kernel_count, size_t kernel_size,
                                              mlsl_data_type dtype, int dist_update)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->AddParameterSet(kernel_count, kernel_size,
                                                                                TO_MLSL(dtype, DataType), (bool)dist_update));
}

int mlsl_operation_reg_info_add_parameter_set_with_compress(mlsl_operation_reg_info reg_info, size_t kernel_count, size_t kernel_size,
                                              mlsl_data_type dtype, int dist_update, mlsl_compression_type compress_type)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->AddParameterSet(kernel_count, kernel_size,
                                                                                TO_MLSL(dtype, DataType), (bool)dist_update, TO_MLSL(compress_type, CompressionType)));
}

int mlsl_operation_reg_info_validate(mlsl_operation_reg_info reg_info, mlsl_distribution dist)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(reg_info, OperationRegInfo*)->Validate(HNDL_TO_MLSL(dist, Distribution*)));
}


/** mlsl_operation */

int mlsl_operation_set_distribution(mlsl_operation op, mlsl_distribution dist)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(op, Operation*)->SetDistribution(HNDL_TO_MLSL(dist, Distribution*)));
}

int mlsl_operation_get_distribution(mlsl_operation op, mlsl_distribution* dist)
{
    TRY_CATCH_RETURN((*dist) = MLSL_TO_HNDL(HNDL_TO_MLSL(op, Operation*)->GetDistribution(), mlsl_distribution));
}

int mlsl_operation_get_session(mlsl_operation op, mlsl_session* session)
{
    TRY_CATCH_RETURN((*session) = MLSL_TO_HNDL(HNDL_TO_MLSL(op, Operation*)->GetSession(), mlsl_session));
}

int mlsl_operation_get_op_type(mlsl_operation op, mlsl_op_type* op_type)
{
    TRY_CATCH_RETURN((*op_type) = TO_CMLSL(HNDL_TO_MLSL(op, Operation*)->GetOpType(), mlsl_op_type));
}

int mlsl_operation_set_prev(mlsl_operation op, mlsl_operation prev, size_t act_idx, size_t prev_op_act_idx)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(op, Operation*)->SetPrev(HNDL_TO_MLSL(prev, Operation*), act_idx, prev_op_act_idx));
}

int mlsl_operation_set_next(mlsl_operation op, mlsl_operation next, size_t act_idx, size_t next_op_act_idx)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(op, Operation*)->SetNext(HNDL_TO_MLSL(next, Operation*), act_idx, next_op_act_idx));
}

int mlsl_operation_get_name(mlsl_operation op, const char** name)
{
    TRY_CATCH_RETURN((*name) = HNDL_TO_MLSL(op, Operation*)->GetName());
}

int mlsl_operation_get_global_minibatch_size(mlsl_operation op, size_t* global_minibatch_size)
{
    TRY_CATCH_RETURN((*global_minibatch_size) = HNDL_TO_MLSL(op, Operation*)->GetGlobalMinibatchSize());
}

int mlsl_operation_get_local_minibatch_size(mlsl_operation op, size_t* local_minibatch_size)
{
    TRY_CATCH_RETURN((*local_minibatch_size) = HNDL_TO_MLSL(op, Operation*)->GetLocalMinibatchSize());
}

int mlsl_operation_get_global_minibatch_offset(mlsl_operation op, size_t* global_minibatch_offset)
{
    TRY_CATCH_RETURN((*global_minibatch_offset) = HNDL_TO_MLSL(op, Operation*)->GetGlobalMinibatchOffset());
}

int mlsl_operation_get_input_count(mlsl_operation op, size_t* input_count)
{
    TRY_CATCH_RETURN((*input_count) = HNDL_TO_MLSL(op, Operation*)->GetInputCount());
}

int mlsl_operation_get_input(mlsl_operation op, size_t input_idx, mlsl_activation* input_act)
{
    TRY_CATCH_RETURN((*input_act) = MLSL_TO_HNDL(HNDL_TO_MLSL(op, Operation*)->GetInput(input_idx), mlsl_activation));
}

int mlsl_operation_get_output_count(mlsl_operation op, size_t* output_count)
{
    TRY_CATCH_RETURN((*output_count) = HNDL_TO_MLSL(op, Operation*)->GetOutputCount());
}

int mlsl_operation_get_output(mlsl_operation op, size_t output_idx, mlsl_activation* output_act)
{
    TRY_CATCH_RETURN((*output_act) = MLSL_TO_HNDL(HNDL_TO_MLSL(op, Operation*)->GetOutput(output_idx), mlsl_activation));
}

int mlsl_operation_has_parameter_sets(mlsl_operation op, int* has_params)
{
    TRY_CATCH_RETURN((*has_params) = (HNDL_TO_MLSL(op, Operation*)->HasParameterSets() ? 1 : 0));
}

int mlsl_operation_get_parameter_set_count(mlsl_operation op, size_t* param_count)
{
    TRY_CATCH_RETURN((*param_count) = HNDL_TO_MLSL(op, Operation*)->GetParameterSetCount());
}

int mlsl_operation_get_parameter_set(mlsl_operation op, size_t param_idx, mlsl_parameter_set* param_set)
{
    TRY_CATCH_RETURN((*param_set) = MLSL_TO_HNDL(HNDL_TO_MLSL(op, Operation*)->GetParameterSet(param_idx), mlsl_parameter_set));
}


/** mlsl_statistics */

int mlsl_statistics_start(mlsl_statistics stat)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(stat, Statistics*)->Start());
}

int mlsl_statistics_stop(mlsl_statistics stat)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(stat, Statistics*)->Stop());
}

int mlsl_statistics_reset(mlsl_statistics stat)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(stat, Statistics*)->Reset());
}

int mlsl_statistics_print(mlsl_statistics stat)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(stat, Statistics*)->Print());
}

int mlsl_statistics_is_started(mlsl_statistics stat, int* is_started)
{
    TRY_CATCH_RETURN((*is_started) = (HNDL_TO_MLSL(stat, Statistics*)->IsStarted() ? 1 : 0));
}

int mlsl_statistics_is_enabled(mlsl_statistics stat, int* is_enabled)
{
    TRY_CATCH_RETURN((*is_enabled) = (HNDL_TO_MLSL(stat, Statistics*)->IsEnabled() ? 1 : 0));
}

int mlsl_statistics_get_isolation_comm_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetIsolationCommCycles(op_idx));
}

int mlsl_statistics_get_comm_size(mlsl_statistics stat, size_t op_idx, size_t* size)
{
    TRY_CATCH_RETURN((*size) = HNDL_TO_MLSL(stat, Statistics*)->GetCommSize(op_idx));
}

int mlsl_statistics_get_comm_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetCommCycles(op_idx));
}

int mlsl_statistics_get_compute_cycles(mlsl_statistics stat, size_t op_idx, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetComputeCycles(op_idx));
}

int mlsl_statistics_get_total_isolation_comm_cycles(mlsl_statistics stat, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetTotalIsolationCommCycles());
}

int mlsl_statistics_get_total_comm_size(mlsl_statistics stat, size_t* size)
{
    TRY_CATCH_RETURN((*size) = HNDL_TO_MLSL(stat, Statistics*)->GetTotalCommSize());
}

int mlsl_statistics_get_total_comm_cycles(mlsl_statistics stat, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetTotalCommCycles());
}

int mlsl_statistics_get_total_compute_cycles(mlsl_statistics stat, unsigned long long* cycles)
{
    TRY_CATCH_RETURN((*cycles) = HNDL_TO_MLSL(stat, Statistics*)->GetTotalComputeCycles());
}


/** mlsl_session */

int mlsl_session_set_global_minibatch_size(mlsl_session session, size_t global_minibatch_size)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(session, Session*)->SetGlobalMinibatchSize(global_minibatch_size));
}

int mlsl_session_get_global_minibatch_size(mlsl_session session, size_t* global_minibatch_size)
{
    TRY_CATCH_RETURN((*global_minibatch_size) = HNDL_TO_MLSL(session, Session*)->GetGlobalMinibatchSize());
}

int mlsl_session_get_phase_type(mlsl_session session, mlsl_phase_type* phase_type)
{
    TRY_CATCH_RETURN((*phase_type) = TO_CMLSL(HNDL_TO_MLSL(session, Session*)->GetGlobalMinibatchSize(), mlsl_phase_type));
}

int mlsl_session_create_operation_reg_info(mlsl_session session, mlsl_op_type op_type, mlsl_operation_reg_info* reg_info)
{
    TRY_CATCH_RETURN((*reg_info) = MLSL_TO_HNDL(HNDL_TO_MLSL(session, Session*)->CreateOperationRegInfo(TO_MLSL(op_type, OpType)), mlsl_operation_reg_info));
}

int mlsl_session_delete_operation_reg_info(mlsl_session session, mlsl_operation_reg_info reg_info)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(session, Session*)->DeleteOperationRegInfo(HNDL_TO_MLSL(reg_info, OperationRegInfo*)));
}

int mlsl_session_add_operation_with_distribution(mlsl_session session, mlsl_operation_reg_info reg_info, mlsl_distribution dist, size_t* op_idx)
{
    TRY_CATCH_RETURN((*op_idx) = HNDL_TO_MLSL(session, Session*)->AddOperation(HNDL_TO_MLSL(reg_info, OperationRegInfo*),
                                                                               HNDL_TO_MLSL(dist, Distribution*)));
}
int mlsl_session_add_operation(mlsl_session session, mlsl_operation_reg_info reg_info, size_t* op_idx)
{
    TRY_CATCH_RETURN((*op_idx) = HNDL_TO_MLSL(session, Session*)->AddOperation(HNDL_TO_MLSL(reg_info, OperationRegInfo*)));
}
int mlsl_session_remove_operations(mlsl_session session)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(session, Session*)->RemoveOperations());
}

int mlsl_session_get_operation_count(mlsl_session session, size_t* op_count)
{
    TRY_CATCH_RETURN((*op_count) = HNDL_TO_MLSL(session, Session*)->GetOperationCount());
}

int mlsl_session_get_operation(mlsl_session session, size_t op_idx, mlsl_operation* op)
{
    TRY_CATCH_RETURN((*op) = MLSL_TO_HNDL(HNDL_TO_MLSL(session, Session*)->GetOperation(op_idx), mlsl_operation));
}

int mlsl_session_commit(mlsl_session session)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(session, Session*)->Commit());
}

int mlsl_session_get_stats(mlsl_session session, mlsl_statistics* stat)
{
    TRY_CATCH_RETURN((*stat) = MLSL_TO_HNDL(HNDL_TO_MLSL(session, Session*)->GetStats(), mlsl_statistics));
}


/** mlsl_environment */

int mlsl_environment_get_env(mlsl_environment* env)
{
    TRY_CATCH_RETURN((*env) = MLSL_TO_HNDL(&(HNDL_TO_MLSL(env, Environment*)->GetEnv()), mlsl_environment));
}

int mlsl_environment_get_version(int* version)
{
    TRY_CATCH_RETURN((*version) = Environment::GetVersion());
}

int mlsl_environment_configure(mlsl_environment env, const char* config)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Configure(config));
}

int mlsl_environment_init(mlsl_environment env, int* argc, char** argv[])
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Init(argc, argv));
}

int mlsl_environment_finalize(mlsl_environment env)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Finalize());
}

int mlsl_environment_is_initialized(mlsl_environment env, int* is_initialized)
{
    TRY_CATCH_RETURN((*is_initialized) = (HNDL_TO_MLSL(env, Environment*)->IsInitialized() ? 1 : 0));
}

int mlsl_environment_get_process_idx(mlsl_environment env, size_t* process_idx)
{
    TRY_CATCH_RETURN((*process_idx) = HNDL_TO_MLSL(env, Environment*)->GetProcessIdx());
}

int mlsl_environment_get_process_count(mlsl_environment env, size_t* process_count)
{
    TRY_CATCH_RETURN((*process_count) = HNDL_TO_MLSL(env, Environment*)->GetProcessCount());
}

int mlsl_environment_create_session(mlsl_environment env, mlsl_phase_type phase_type, mlsl_session* session)
{
    TRY_CATCH_RETURN((*session) = MLSL_TO_HNDL(HNDL_TO_MLSL(env, Environment*)->CreateSession(TO_MLSL(phase_type, PhaseType)), mlsl_session));
}

int mlsl_environment_delete_session(mlsl_environment env, mlsl_session session)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->DeleteSession(HNDL_TO_MLSL(session, Session*)));
}

int mlsl_environment_create_distribution(mlsl_environment env, size_t data_partitions, size_t model_partitions, mlsl_distribution* dist)
{
    TRY_CATCH_RETURN((*dist) = MLSL_TO_HNDL(HNDL_TO_MLSL(env, Environment*)->CreateDistribution(data_partitions,
                                                                                                model_partitions), mlsl_distribution));
}

int mlsl_environment_delete_distribution(mlsl_environment env, mlsl_distribution dist)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->DeleteDistribution(HNDL_TO_MLSL(dist, Distribution*)));
}

int mlsl_environment_wait(mlsl_environment env, mlsl_comm_req req)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Wait(HNDL_TO_MLSL(req, CommReq*)));
}

int mlsl_environment_test(mlsl_environment env, mlsl_comm_req req, int* is_completed)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Test(HNDL_TO_MLSL(req, CommReq*), (bool*)is_completed));
}

int mlsl_environment_alloc(mlsl_environment env, size_t size, size_t alignment, void** ptr)
{
    TRY_CATCH_RETURN((*ptr) = HNDL_TO_MLSL(env, Environment*)->Alloc(size, alignment));
}

int mlsl_environment_free(mlsl_environment env, void* ptr)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->Free(ptr));
}
int mlsl_environment_set_quantization_params(mlsl_environment env, mlsl_quant_params* params)
{
    TRY_CATCH_RETURN(HNDL_TO_MLSL(env, Environment*)->SetQuantizationParams(HNDL_TO_MLSL(params, QuantParams*)));
}

int mlsl_environment_get_quantization_params(mlsl_environment env, mlsl_quant_params* params)
{
    TRY_CATCH_RETURN((*params) = MLSL_TO_HNDL(HNDL_TO_MLSL(env, Environment*)->GetQuantizationParams(), mlsl_quant_params));
}
