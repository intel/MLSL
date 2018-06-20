#
# Copyright 2016-2018 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ctypes import c_int, c_ulonglong, c_size_t, \
    c_void_p, c_char_p, byref, \
    POINTER, pointer, cdll
import os
import sys
import logging


mlsl_module = sys.modules[__name__]
mlsl_module._ref_count = 0
mlsl_module._log_level = 0
mlsl_module._mlsl_obj = None


class _CommBlockInfo(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_mb_offset(self):
        mb_offset = c_size_t()
        MLSL._comm_block_info_get_mb_offset_fn(self.__handle, byref(mb_offset))
        return mb_offset.value

    def get_mb_count(self):
        mb_count = c_size_t()
        MLSL._comm_block_info_get_mb_count_fn(self.__handle, byref(mb_count))
        return mb_count.value

    def get_fm_offset(self):
        fm_offset = c_size_t()
        MLSL._comm_block_info_get_fm_offset_fn(self.__handle, byref(fm_offset))
        return fm_offset.value

    def get_fm_count(self):
        fm_count = c_size_t()
        MLSL._comm_block_info_get_fm_count_fn(self.__handle, byref(fm_count))
        return fm_count.value

    def get_fm_size(self):
        fm_size = c_size_t()
        MLSL._comm_block_info_get_fm_size_fn(self.__handle, byref(fm_size))
        return fm_size.value

    def get_data_type(self):
        data_type = c_int()
        MLSL._comm_block_info_get_data_type_fn(self.__handle, byref(data_type))
        return data_type.value

    def get_buf_offset(self):
        buf_offset = c_size_t()
        MLSL._comm_block_info_get_buf_offset_fn(self.__handle, byref(buf_offset))
        return buf_offset.value


class _Activation(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_global_fm_count(self):
        fm_count = c_size_t()
        MLSL._activation_get_global_fm_count_fn(self.__handle, byref(fm_count))
        return fm_count.value

    def get_global_fm_offset(self):
        fm_offset = c_size_t()
        MLSL._activation_get_global_fm_offset_fn(self.__handle, byref(fm_offset))
        return fm_offset.value

    def get_local_fm_count(self):
        fm_count = c_size_t()
        MLSL._activation_get_local_fm_count_fn(self.__handle, byref(fm_count))
        return fm_count.value

    def get_pack_block_count(self):
        block_count = c_size_t()
        MLSL._activation_get_pack_block_count_fn(self.__handle, byref(block_count))
        return block_count.value

    def get_unpack_block_count(self):
        block_count = c_size_t()
        MLSL._activation_get_unpack_block_count_fn(self.__handle, byref(block_count))
        return block_count.value

    def get_pack_block(self, block_idx):
        block_handle = c_ulonglong()
        MLSL._activation_get_pack_block_fn(self.__handle, block_idx, byref(block_handle))
        return _CommBlockInfo(block_handle)

    def get_unpack_block(self, block_idx):
        block_handle = c_ulonglong()
        MLSL._activation_get_unpack_block_fn(self.__handle, block_idx, byref(block_handle))
        return _CommBlockInfo(block_handle)

    def get_data_type(self):
        data_type = c_int()
        MLSL._activation_get_data_type_fn(self.__handle, byref(data_type))
        return data_type.value

    def get_fm_size(self):
        fm_size = c_size_t()
        MLSL._activation_get_fm_size_fn(self.__handle, byref(fm_size))
        return fm_size.value

    def get_comm_buf(self):
        comm_buf = c_void_p()
        MLSL._activation_get_comm_buf_fn(self.__handle, byref(comm_buf))
        return comm_buf

    def get_comm_buf_size(self):
        size = c_size_t()
        MLSL._activation_get_comm_buf_size_fn(self.__handle, byref(size))
        return size.value

    def start_comm(self, buf):
        MLSL._activation_start_comm_fn(self.__handle, buf)

    def wait_comm(self):
        ret_buf = c_void_p()
        MLSL._activation_wait_comm_fn(self.__handle, byref(ret_buf))
        return ret_buf


class _ParameterSet(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_global_kernel_count(self):
        kernel_count = c_size_t()
        MLSL._parameter_set_get_global_kernel_count_fn(self.__handle, byref(kernel_count))
        return kernel_count.value

    def get_global_kernel_offset(self):
        kernel_offset = c_size_t()
        MLSL._parameter_set_get_global_kernel_offset_fn(self.__handle, byref(kernel_offset))
        return kernel_offset.value

    def get_local_kernel_count(self):
        kernel_count = c_size_t()
        MLSL._parameter_set_get_local_kernel_count_fn(self.__handle, byref(kernel_count))
        return kernel_count.value

    def get_owned_kernel_count(self):
        kernel_count = c_size_t()
        MLSL._parameter_set_get_owned_kernel_count_fn(self.__handle, byref(kernel_count))
        return kernel_count.value

    def get_owned_kernel_offset(self):
        kernel_offset = c_size_t()
        MLSL._parameter_set_get_owned_kernel_offset_fn(self.__handle, byref(kernel_offset))
        return kernel_offset.value

    def get_data_type(self):
        data_type = c_int()
        MLSL._parameter_set_get_data_type_fn(self.__handle, byref(data_type))
        return data_type.value

    def get_kernel_size(self):
        kernel_size = c_size_t()
        MLSL._parameter_set_get_kernel_size_fn(self.__handle, byref(kernel_size))
        return kernel_size.value

    def is_distributed_update(self):
        is_dist_update = c_int()
        MLSL._parameter_set_get_kernel_size_fn(self.__handle, byref(is_dist_update))
        return True if is_dist_update.value == 1 else False

    def start_gradient_comm(self, buf):
        MLSL._parameter_set_start_gradient_comm_fn(self.__handle, buf)

    def wait_gradient_comm(self):
        ret_buf = c_void_p()
        MLSL._parameter_set_wait_gradient_comm_fn(self.__handle, byref(ret_buf))
        return ret_buf if ret_buf.value else None

    def test_gradient_comm(self):
        ret_buf = c_void_p()
        is_completed = c_int()
        MLSL._parameter_set_test_gradient_comm_fn(self.__handle, byref(is_completed),
                                                  byref(ret_buf))
        return (ret_buf if ret_buf.value else None), (True if is_completed.value == 1 else False)

    def start_increment_comm(self, buf):
        MLSL._parameter_set_start_increment_comm_fn(self.__handle, buf)

    def wait_increment_comm(self):
        ret_buf = c_void_p()
        MLSL._parameter_set_wait_increment_comm_fn(self.__handle, byref(ret_buf))
        return ret_buf if ret_buf.value else None


class _Distribution(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_handle(self):
        return self.__handle

    def get_process_count(self, group_type):
        process_count = c_size_t()
        MLSL._distribution_get_process_count_fn(self.__handle, group_type, byref(process_count))
        return process_count.value

    def get_process_idx(self, group_type):
        process_idx = c_size_t()
        MLSL._distribution_get_process_idx_fn(self.__handle, group_type, byref(process_idx))
        return process_idx.value

    def bcast(self, buf, count, data_type, root_idx, group_type):
        req = c_ulonglong()
        MLSL._distribution_bcast_fn(self.__handle, buf, count, data_type,
                                    root_idx, group_type, byref(req))
        return req

    def reduce(self, send_buf, recv_buf, count, data_type, red_type, root_idx, group_type):
        req = c_ulonglong()
        MLSL._distribution_reduce_fn(self.__handle, send_buf, recv_buf, count, data_type, red_type,
                                     root_idx, group_type, byref(req))
        return req

    def all_reduce(self, send_buf, recv_buf, count, data_type, red_type, group_type):
        req = c_ulonglong()
        MLSL._distribution_all_reduce_fn(self.__handle, send_buf, recv_buf, count, data_type,
                                         red_type, group_type, byref(req))
        return req

    def all_to_all(self, send_buf, send_count, recv_buf, data_type, group_type):
        req = c_ulonglong()
        MLSL._distribution_all_to_all_fn(self.__handle, send_buf, send_count, recv_buf,
                                         data_type, group_type, byref(req))
        return req

    def all_to_allv(self, send_buf, send_counts, send_offsets, recv_buf, recv_counts, recv_offsets, data_type, group_type):
        req = c_ulonglong()
        MLSL._distribution_all_to_allv_fn(self.__handle, send_buf, send_counts, send_offsets, recv_buf,
                                         recv_counts, recv_offsets, data_type, group_type, byref(req))
        return req

    def gather(self, send_buf, send_count, recv_buf, data_type, root_idx, group_type):
        req = c_ulonglong()
        MLSL._distribution_gather_fn(self.__handle, send_buf, send_count, recv_buf, data_type,
                                     root_idx, group_type, byref(req))
        return req

    def all_gather(self, send_buf, send_count, recv_buf, data_type, group_type):
        req = c_ulonglong()
        MLSL._distribution_all_gather_fn(self.__handle, send_buf, send_count, recv_buf,
                                         data_type, group_type, byref(req))
        return req

    def scatter(self, send_buf, recv_buf, recv_count, data_type, root_idx, group_type):
        req = c_ulonglong()
        MLSL._distribution_scatter_fn(self.__handle, send_buf, recv_buf, recv_count, data_type,
                                      root_idx, group_type, byref(req))
        return req

    def reduce_scatter(self, send_buf, recv_buf, recv_count, data_type, red_type, group_type):
        req = c_ulonglong()
        MLSL._distribution_reduce_scatter_fn(self.__handle, send_buf, recv_buf, recv_count,
                                             data_type, red_type, group_type, byref(req))
        return req

    def barrier(self, group_type):
        MLSL._distribution_barrier_fn(self.__handle, group_type)


class _OperationRegInfo(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_handle(self):
        return self.__handle

    def set_name(self, name):
        MLSL._operation_reg_info_set_name_fn(self.__handle, c_char_p(name))

    def add_input(self, fm_count, fm_size, data_type):
        MLSL._operation_reg_info_add_input_fn(self.__handle, fm_count, fm_size, data_type)

    def add_output(self, fm_count, fm_size, data_type):
        MLSL._operation_reg_info_add_output_fn(self.__handle, fm_count, fm_size, data_type)

    def add_parameter_set(self, kernel_count, kernel_size, data_type, dist_update):
        MLSL._operation_reg_info_add_parameter_set_fn(self.__handle, kernel_count, kernel_size,
                                                      data_type, dist_update)

    def validate(self, dist):
        MLSL._operation_reg_info_validate_fn(self.__handle, dist.get_handle())


class _Operation(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_handle(self):
        return self.__handle

    def set_distribution(self, dist):
        MLSL._operation_set_distribution_fn(self.__handle, dist.get_handle())

    def get_distribution(self):
        dist_handle = c_ulonglong()
        MLSL._operation_get_distribution_fn(self.__handle, byref(dist_handle))
        return _Distribution(dist_handle)

    def get_session(self):
        _session_handle = c_ulonglong()
        MLSL._operation_get_session_fn(self.__handle, byref(_session_handle))
        return _Session(_session_handle)

    def get_op_type(self):
        op_type = c_int()
        MLSL._operation_get_op_type_fn(self.__handle, byref(op_type))
        return op_type.value

    def set_prev(self, prev_op, act_idx, prev_op_act_idx):
        MLSL._operation_set_prev_fn(self.__handle, prev_op.get_handle(), act_idx, prev_op_act_idx)

    def set_next(self, next_op, act_idx, next_op_act_idx):
        MLSL._operation_set_next_fn(self.__handle, next_op.get_handle(), act_idx, next_op_act_idx)

    def get_name(self):
        name = c_char_p()
        MLSL._operation_get_name_fn(self.__handle, byref(name))
        return name.value

    def get_global_minibatch_size(self):
        mb_size = c_size_t()
        MLSL._operation_get_global_minibatch_size_fn(self.__handle, byref(mb_size))
        return mb_size.value

    def get_local_minibatch_size(self):
        mb_size = c_size_t()
        MLSL._operation_get_local_minibatch_size_fn(self.__handle, byref(mb_size))
        return mb_size.value

    def get_global_minibatch_offset(self):
        mb_offset = c_size_t()
        MLSL._operation_get_global_minibatch_offset_fn(self.__handle, byref(mb_offset))
        return mb_offset.value

    def get_input_count(self):
        count = c_size_t()
        MLSL._operation_get_input_count_fn(self.__handle, byref(count))
        return count.value

    def get_input(self, idx):
        handle = c_ulonglong()
        MLSL._operation_get_input_fn(self.__handle, idx, byref(handle))
        return _Activation(handle)

    def get_output_count(self):
        count = c_size_t()
        MLSL._operation_get_output_count_fn(self.__handle, byref(count))
        return count.value

    def get_output(self, idx):
        handle = c_ulonglong()
        MLSL._operation_get_output_fn(self.__handle, idx, byref(handle))
        return _Activation(handle)

    def has_parameter_sets(self):
        has_param_sets = c_int()
        MLSL._operation_has_parameter_sets_fn(self.__handle, byref(has_param_sets))
        return True if has_param_sets.value == 1 else False

    def get_parameter_set_count(self):
        count = c_size_t()
        MLSL._operation_get_parameter_set_count_fn(self.__handle, byref(count))
        return count.value

    def get_parameter_set(self, idx):
        handle = c_ulonglong()
        MLSL._operation_get_parameter_set_fn(self.__handle, idx, byref(handle))
        return _ParameterSet(handle)


class _Statistics(object):
    def __init__(self, handle):
        self.__handle = handle

    def start(self):
        MLSL._statistics_start_fn(self.__handle)

    def stop(self):
        MLSL._statistics_stop_fn(self.__handle)

    def reset(self):
        MLSL._statistics_reset_fn(self.__handle)

    def dump(self):
        MLSL._statistics_print_fn(self.__handle)

    def is_started(self):
        is_started = c_int()
        MLSL._statistics_is_started_fn(self.__handle, byref(is_started))
        return True if is_started.value == 1 else False

    def is_enabled(self):
        is_enabled = c_int()
        MLSL._statistics_is_enabled_fn(self.__handle, byref(is_enabled))
        return True if is_enabled.value == 1 else False

    def get_isolation_comm_cycles(self, op_idx):
        cycles = c_ulonglong()
        MLSL._statistics_get_isolation_comm_cycles_fn(self.__handle, op_idx, byref(cycles))
        return cycles.value

    def get_comm_size(self, op_idx):
        size = c_size_t()
        MLSL._statistics_get_comm_size_fn(self.__handle, op_idx, byref(size))
        return size.value

    def get_comm_cycles(self, op_idx):
        cycles = c_ulonglong()
        MLSL._statistics_get_comm_cycles_fn(self.__handle, op_idx, byref(cycles))
        return cycles.value

    def get_compute_cycles(self, op_idx):
        cycles = c_ulonglong()
        MLSL._statistics_get_compute_cycles_fn(self.__handle, op_idx, byref(cycles))
        return cycles.value

    def get_total_isolation_comm_cycles(self):
        cycles = c_ulonglong()
        MLSL._statistics_get_total_isolation_comm_cycles_fn(self.__handle, byref(cycles))
        return cycles.value

    def get_total_comm_size(self):
        size = c_size_t()
        MLSL._statistics_get_total_comm_size_fn(self.__handle, byref(size))
        return size.value

    def get_total_comm_cycles(self):
        cycles = c_ulonglong()
        MLSL._statistics_get_total_comm_cycles_fn(self.__handle, byref(cycles))
        return cycles.value

    def get_total_compute_cycles(self):
        cycles = c_ulonglong()
        MLSL._statistics_get_total_compute_cycles_fn(self.__handle, byref(cycles))
        return cycles.value


class _Session(object):
    def __init__(self, handle):
        self.__handle = handle

    def get_handle(self):
        return self.__handle

    def set_global_minibatch_size(self, batch_size):
        MLSL._session_set_global_minibatch_size_fn(self.__handle, batch_size)

    def get_global_minibatch_size(self):
        batch_size = c_size_t()
        MLSL._session_get_global_minibatch_size_fn(self.__handle, byref(batch_size))
        return batch_size.value

    def get_phase_type(self):
        batch_size = c_int()
        MLSL._session_get_phase_type_fn(self.__handle, byref(batch_size))
        return batch_size.value

    def create_operation_reg_info(self, op_type):
        reg_info_handle = c_ulonglong()
        MLSL._session_create_operation_reg_info_fn(self.__handle, op_type, byref(reg_info_handle))
        return _OperationRegInfo(reg_info_handle)

    def delete_operation_reg_info(self, reg_info):
        MLSL._session_delete_operation_reg_info_fn(self.__handle, reg_info.get_handle())

    def add_operation_with_distribution(self, reg_info, dist):
        op_idx = c_size_t()
        MLSL._session_add_operation_with_distribution_fn(self.__handle, reg_info.get_handle(),
                                                         dist.get_handle(), byref(op_idx))
        return op_idx.value

    def add_operation(self, reg_info):
        op_idx = c_size_t()
        MLSL._session_add_operation_fn(self.__handle, reg_info.get_handle(), byref(op_idx))
        return op_idx.value

    def remove_operations(self):
        MLSL._session_remove_operations_fn(self.__handle)

    def get_operation_count(self):
        count = c_size_t()
        MLSL._session_get_operation_count_fn(self.__handle, byref(count))
        return count.value

    def get_operation(self, op_idx):
        op_handle = c_ulonglong()
        MLSL._session_get_operation_fn(self.__handle, op_idx, byref(op_handle))
        return _Operation(op_handle)

    def commit(self):
        MLSL._session_commit_fn(self.__handle)

    def get_stats(self):
        stat_handle = c_ulonglong()
        MLSL._session_get_stats_fn(self.__handle, byref(stat_handle))
        return _Statistics(stat_handle)


class DataType(object):
    FLOAT = 0
    DOUBLE = 1


class PhaseType(object):
    TRAIN = 0
    TEST = 1


class GroupType(object):
    DATA = 0
    MODEL = 1
    GLOBAL = 2


class ReductionType(object):
    SUM = 0
    MIN = 1
    MAX = 2


class OperationType(object):
    CC = 0
    BIAS = 1
    ACT = 2
    POOL = 3
    SPLIT = 4
    CONCAT = 5
    BCAST = 6
    REDUCE = 7
    DATA = 8
    EVAL = 9


class MLSL(object):

    __handle = c_ulonglong()
    __dll = None
    __is_loaded = False
    __is_initialized = False

    _comm_block_info_get_mb_offset_fn = None
    _comm_block_info_get_mb_count_fn = None
    _comm_block_info_get_fm_offset_fn = None
    _comm_block_info_get_fm_count_fn = None
    _comm_block_info_get_fm_size_fn = None
    _comm_block_info_get_data_type_fn = None
    _comm_block_info_get_buf_offset_fn = None
    _activation_get_global_fm_count_fn = None
    _activation_get_global_fm_offset_fn = None
    _activation_get_local_fm_count_fn = None
    _activation_get_pack_block_count_fn = None
    _activation_get_unpack_block_count_fn = None
    _activation_get_pack_block_fn = None
    _activation_get_unpack_block_fn = None
    _activation_get_data_type_fn = None
    _activation_get_fm_size_fn = None
    _activation_get_comm_buf_fn = None
    _activation_get_comm_buf_size_fn = None
    _activation_start_comm_fn = None
    _activation_wait_comm_fn = None
    _parameter_set_get_global_kernel_count_fn = None
    _parameter_set_get_global_kernel_offset_fn = None
    _parameter_set_get_local_kernel_count_fn = None
    _parameter_set_get_owned_kernel_count_fn = None
    _parameter_set_get_owned_kernel_offset_fn = None
    _parameter_set_get_data_type_fn = None
    _parameter_set_get_kernel_size_fn = None
    _parameter_set_is_distributed_update_fn = None
    _parameter_set_start_gradient_comm_fn = None
    _parameter_set_start_increment_comm_fn = None
    _parameter_set_wait_gradient_comm_fn = None
    _parameter_set_test_gradient_comm_fn = None
    _parameter_set_wait_increment_comm_fn = None
    _distribution_get_process_count_fn = None
    _distribution_get_process_idx_fn = None
    _distribution_bcast_fn = None
    _distribution_reduce_fn = None
    _distribution_all_reduce_fn = None
    _distribution_all_to_all_fn = None
    _distribution_all_to_allv_fn = None
    _distribution_gather_fn = None
    _distribution_all_gather_fn = None
    _distribution_scatter_fn = None
    _distribution_reduce_scatter_fn = None
    _distribution_barrier_fn = None
    _operation_reg_info_set_name_fn = None
    _operation_reg_info_add_input_fn = None
    _operation_reg_info_add_output_fn = None
    _operation_reg_info_add_parameter_set_fn = None
    _operation_reg_info_validate_fn = None
    _operation_set_distribution_fn = None
    _operation_get_distribution_fn = None
    _operation_get_session_fn = None
    _operation_get_op_type_fn = None
    _operation_set_prev_fn = None
    _operation_set_next_fn = None
    _operation_get_name_fn = None
    _operation_get_global_minibatch_size_fn = None
    _operation_get_local_minibatch_size_fn = None
    _operation_get_global_minibatch_offset_fn = None
    _operation_get_input_count_fn = None
    _operation_get_input_fn = None
    _operation_get_output_count_fn = None
    _operation_get_output_fn = None
    _operation_has_parameter_sets_fn = None
    _operation_get_parameter_set_count_fn = None
    _operation_get_parameter_set_fn = None
    _statistics_start_fn = None
    _statistics_stop_fn = None
    _statistics_reset_fn = None
    _statistics_is_started_fn = None
    _statistics_is_enabled_fn = None
    _statistics_get_isolation_comm_cycles_fn = None
    _statistics_get_comm_size_fn = None
    _statistics_get_comm_cycles_fn = None
    _statistics_get_compute_cycles_fn = None
    _statistics_get_total_isolation_comm_cycles_fn = None
    _statistics_get_total_comm_size_fn = None
    _statistics_get_total_comm_cycles_fn = None
    _statistics_get_total_compute_cycles_fn = None
    _session_set_global_minibatch_size_fn = None
    _session_get_global_minibatch_size_fn = None
    _session_get_phase_type_fn = None
    _session_create_operation_reg_info_fn = None
    _session_delete_operation_reg_info_fn = None
    _session_add_operation_with_distribution_fn = None
    _session_add_operation_fn = None
    _session_remove_operations_fn = None
    _session_get_operation_count_fn = None
    _session_get_operation_fn = None
    _session_commit_fn = None
    _session_get_stats_fn = None
    _environment_get_env_fn = None
    _environment_get_version_fn = None
    _environment_configure_fn = None
    _environment_init_fn = None
    _environment_finalize_fn = None
    _environment_is_initialized_fn = None
    _environment_get_process_idx_fn = None
    _environment_get_process_count_fn = None
    _environment_create_session_fn = None
    _environment_delete_session_fn = None
    _environment_create_distribution_fn = None
    _environment_delete_distribution_fn = None
    _environment_wait_fn = None
    _environment_test_fn = None
    _environment_alloc_fn = None
    _environment_free_fn = None

    def get_version(self):
        version = c_int()
        MLSL._environment_get_version_fn(MLSL.__handle, byref(version))
        return version.value

    def configure(self, config):
        MLSL._environment_configure_fn(MLSL.__handle, c_char_p(config))

    def init(self):
        mlsl_module._ref_count += 1
        logging.debug("init: ref_count: %d", mlsl_module._ref_count)
        is_mlsl_initialized = c_int()
        MLSL._environment_is_initialized_fn(MLSL.__handle, byref(is_mlsl_initialized))
        if is_mlsl_initialized.value == 0:
            MLSL._environment_init_fn(MLSL.__handle, byref(c_int(0)), byref(pointer(c_char_p())))
            logging.debug("Intel(R) MLSL is initialized")
        else:
            logging.debug("Intel(R) MLSL is already initialized")
            pass

    def finalize(self):
        mlsl_module._ref_count -= 1
        logging.debug("finalize: ref_count: %d", mlsl_module._ref_count)
        if mlsl_module._ref_count == 0:
            is_mlsl_initialized = c_int()
            MLSL._environment_is_initialized_fn(MLSL.__handle, byref(is_mlsl_initialized))
            if is_mlsl_initialized.value == 1:
                MLSL._environment_finalize_fn(MLSL.__handle)
                logging.debug("Intel(R) MLSL is finalized")
            else:
                logging.debug("Intel(R) MLSL is already finalized")
                pass

    def is_initialized(self):
        is_initialized = c_int()
        MLSL._environment_is_initialized_fn(MLSL.__handle, byref(is_initialized))
        return True if is_initialized.value == 1 else False

    def get_process_idx(self):
        process_idx = c_size_t()
        MLSL._environment_get_process_idx_fn(MLSL.__handle, byref(process_idx))
        return process_idx.value

    def get_process_count(self):
        process_count = c_size_t()
        MLSL._environment_get_process_count_fn(MLSL.__handle, byref(process_count))
        return process_count.value

    def create_session(self, phase_type):
        _session_handle = c_ulonglong()
        MLSL._environment_create_session_fn(MLSL.__handle, phase_type, byref(_session_handle))
        return _Session(_session_handle)

    def delete_session(self, session):
        MLSL._environment_delete_session_fn(MLSL.__handle, session.get_handle())

    def create_distribution(self, data_parts, model_parts):
        dist_handle = c_ulonglong()
        MLSL._environment_create_distribution_fn(MLSL.__handle, data_parts, model_parts,
                                                 byref(dist_handle))
        return _Distribution(dist_handle)

    def delete_distribution(self, dist):
        MLSL._environment_delete_distribution_fn(MLSL.__handle, dist.get_handle())

    def wait(self, req):
        MLSL._environment_wait_fn(MLSL.__handle, req)

    def test(self, req):
        is_completed = c_int()
        MLSL._environment_test_fn(MLSL.__handle, req, byref(is_completed))
        return True if is_completed.value == 1 else False

    def alloc(self, size, alignment):
        ptr = c_void_p()
        MLSL._environment_alloc_fn(MLSL.__handle, size, alignment, byref(ptr))
        return ptr

    def free(self, ptr):
        MLSL._environment_free_fn(MLSL.__handle, ptr)

    def __init__(self):
        super(MLSL, self).__init__()
        mlsl_path = None
        if MLSL.__is_loaded is False:
            try:
                log_level = os.getenv("MLSL_LOG_LEVEL")
                if log_level is not None:
                    mlsl_module._log_level = int(log_level)

                root_path = os.getenv("MLSL_ROOT", "")
                mlsl_path = os.path.join(root_path, "intel64/lib/libmlsl.so")
                logging.debug("mlsl_path: %s", mlsl_path)
                MLSL.__dll = cdll.LoadLibrary(mlsl_path)
                logging.debug("Intel(R) MLSL library is loaded from %s", mlsl_path)
                MLSL.__is_loaded = True
            except:
                logging.error("Could not load Intel(R) MLSL library: %s", mlsl_path)
                logging.error("Error: %s", sys.exc_info()[0])
                raise

        if MLSL.__is_loaded is True and MLSL.__is_initialized is False:
            MLSL.__dll.mlsl_environment_get_env(byref(MLSL.__handle))

            # _CommBlockInfo
            MLSL._comm_block_info_get_mb_offset_fn = MLSL.__dll.mlsl_comm_block_info_get_mb_offset
            MLSL._comm_block_info_get_mb_offset_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_mb_count_fn = MLSL.__dll.mlsl_comm_block_info_get_mb_count
            MLSL._comm_block_info_get_mb_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_fm_offset_fn = MLSL.__dll.mlsl_comm_block_info_get_fm_offset
            MLSL._comm_block_info_get_fm_offset_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_fm_count_fn = MLSL.__dll.mlsl_comm_block_info_get_fm_count
            MLSL._comm_block_info_get_fm_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_fm_size_fn = MLSL.__dll.mlsl_comm_block_info_get_fm_size
            MLSL._comm_block_info_get_fm_size_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_data_type_fn = MLSL.__dll.mlsl_comm_block_info_get_data_type
            MLSL._comm_block_info_get_data_type_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._comm_block_info_get_buf_offset_fn = \
                MLSL.__dll.mlsl_comm_block_info_get_buf_offset
            MLSL._comm_block_info_get_buf_offset_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            # _Activation
            MLSL._activation_get_global_fm_count_fn = \
                MLSL.__dll.mlsl_activation_get_global_fm_count
            MLSL._activation_get_global_fm_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_global_fm_offset_fn = \
                MLSL.__dll.mlsl_activation_get_global_fm_offset
            MLSL._activation_get_global_fm_offset_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_local_fm_count_fn = MLSL.__dll.mlsl_activation_get_local_fm_count
            MLSL._activation_get_local_fm_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_pack_block_count_fn = \
                MLSL.__dll.mlsl_activation_get_pack_block_count
            MLSL._activation_get_pack_block_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_unpack_block_count_fn = \
                MLSL.__dll.mlsl_activation_get_unpack_block_count
            MLSL._activation_get_unpack_block_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_pack_block_fn = MLSL.__dll.mlsl_activation_get_pack_block
            MLSL._activation_get_pack_block_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._activation_get_unpack_block_fn = MLSL.__dll.mlsl_activation_get_unpack_block
            MLSL._activation_get_unpack_block_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._activation_get_data_type_fn = MLSL.__dll.mlsl_activation_get_data_type
            MLSL._activation_get_data_type_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._activation_get_fm_size_fn = MLSL.__dll.mlsl_activation_get_fm_size
            MLSL._activation_get_fm_size_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_get_comm_buf_fn = MLSL.__dll.mlsl_activation_get_comm_buf
            MLSL._activation_get_comm_buf_fn.argtypes = [c_ulonglong, POINTER(c_void_p)]

            MLSL._activation_get_comm_buf_size_fn = MLSL.__dll.mlsl_activation_get_comm_buf_size
            MLSL._activation_get_comm_buf_size_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._activation_start_comm_fn = MLSL.__dll.mlsl_activation_start_comm
            MLSL._activation_start_comm_fn.argtypes = [c_ulonglong, c_void_p]

            MLSL._activation_wait_comm_fn = MLSL.__dll.mlsl_activation_wait_comm
            MLSL._activation_wait_comm_fn.argtypes = [c_ulonglong, POINTER(c_void_p)]

            # _ParameterSet
            MLSL._parameter_set_get_global_kernel_count_fn = \
                MLSL.__dll.mlsl_parameter_set_get_global_kernel_count
            MLSL._parameter_set_get_global_kernel_count_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_get_global_kernel_offset_fn = \
                MLSL.__dll.mlsl_parameter_set_get_global_kernel_offset
            MLSL._parameter_set_get_global_kernel_offset_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_get_local_kernel_count_fn = \
                MLSL.__dll.mlsl_parameter_set_get_local_kernel_count
            MLSL._parameter_set_get_local_kernel_count_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_get_owned_kernel_count_fn = \
                MLSL.__dll.mlsl_parameter_set_get_owned_kernel_count
            MLSL._parameter_set_get_owned_kernel_count_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_get_owned_kernel_offset_fn = \
                MLSL.__dll.mlsl_parameter_set_get_owned_kernel_offset
            MLSL._parameter_set_get_owned_kernel_offset_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_get_data_type_fn = MLSL.__dll.mlsl_parameter_set_get_data_type
            MLSL._parameter_set_get_data_type_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._parameter_set_get_kernel_size_fn = MLSL.__dll.mlsl_parameter_set_get_kernel_size
            MLSL._parameter_set_get_kernel_size_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._parameter_set_is_distributed_update_fn = \
                MLSL.__dll.mlsl_parameter_set_is_distributed_update
            MLSL._parameter_set_is_distributed_update_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._parameter_set_start_gradient_comm_fn = \
                MLSL.__dll.mlsl_parameter_set_start_gradient_comm
            MLSL._parameter_set_start_gradient_comm_fn.argtypes = [c_ulonglong, c_void_p]

            MLSL._parameter_set_start_increment_comm_fn = \
                MLSL.__dll.mlsl_parameter_set_start_increment_comm
            MLSL._parameter_set_start_increment_comm_fn.argtypes = [c_ulonglong, c_void_p]

            MLSL._parameter_set_wait_gradient_comm_fn = \
                MLSL.__dll.mlsl_parameter_set_wait_gradient_comm
            MLSL._parameter_set_wait_gradient_comm_fn.argtypes = [c_ulonglong, POINTER(c_void_p)]

            MLSL._parameter_set_test_gradient_comm_fn = \
                MLSL.__dll.mlsl_parameter_set_test_gradient_comm
            MLSL._parameter_set_test_gradient_comm_fn.argtypes = \
                [c_ulonglong, POINTER(c_int), POINTER(c_void_p)]

            MLSL._parameter_set_wait_increment_comm_fn = \
                MLSL.__dll.mlsl_parameter_set_wait_increment_comm
            MLSL._parameter_set_wait_increment_comm_fn.argtypes = [c_ulonglong, POINTER(c_void_p)]

            # _Distribution
            MLSL._distribution_get_process_count_fn = \
                MLSL.__dll.mlsl_distribution_get_process_count
            MLSL._distribution_get_process_count_fn.argtypes = \
                [c_ulonglong, c_int, POINTER(c_size_t)]

            MLSL._distribution_get_process_idx_fn = MLSL.__dll.mlsl_distribution_get_process_idx
            MLSL._distribution_get_process_idx_fn.argtypes = \
                [c_ulonglong, c_int, POINTER(c_size_t)]

            MLSL._distribution_bcast_fn = MLSL.__dll.mlsl_distribution_bcast
            MLSL._distribution_bcast_fn.argtypes = \
                [c_ulonglong, c_void_p, c_size_t, c_int, c_size_t, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_reduce_fn = MLSL.__dll.mlsl_distribution_reduce
            MLSL._distribution_reduce_fn.argtypes = \
                [c_ulonglong, c_void_p, c_void_p, c_size_t, c_int, c_int, c_size_t, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_all_reduce_fn = MLSL.__dll.mlsl_distribution_all_reduce
            MLSL._distribution_all_reduce_fn.argtypes = \
                [c_ulonglong, c_void_p, c_void_p, c_size_t, c_int, c_int, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_all_to_all_fn = MLSL.__dll.mlsl_distribution_all_to_all
            MLSL._distribution_all_to_all_fn.argtypes = \
                [c_ulonglong, c_void_p, c_size_t, c_void_p, c_int, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_all_to_allv_fn = MLSL.__dll.mlsl_distribution_all_to_allv
            MLSL._distribution_all_to_allv_fn.argtypes = \
                [c_ulonglong, c_void_p, POINTER(c_size_t), POINTER(c_size_t), c_void_p, POINTER(c_size_t), POINTER(c_size_t), c_int, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_gather_fn = MLSL.__dll.mlsl_distribution_gather
            MLSL._distribution_gather_fn.argtypes = \
                [c_ulonglong, c_void_p, c_size_t, c_void_p, c_int, c_size_t, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_all_gather_fn = MLSL.__dll.mlsl_distribution_all_gather
            MLSL._distribution_all_gather_fn.argtypes = \
                [c_ulonglong, c_void_p, c_size_t, c_void_p, c_int, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_scatter_fn = MLSL.__dll.mlsl_distribution_scatter
            MLSL._distribution_scatter_fn.argtypes = \
                [c_ulonglong, c_void_p, c_void_p, c_size_t, c_int, c_size_t, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_reduce_scatter_fn = MLSL.__dll.mlsl_distribution_reduce_scatter
            MLSL._distribution_reduce_scatter_fn.argtypes = \
                [c_ulonglong, c_void_p, c_void_p, c_size_t, c_int, c_int, c_int,
                 POINTER(c_ulonglong)]

            MLSL._distribution_barrier_fn = MLSL.__dll.mlsl_distribution_barrier
            MLSL._distribution_barrier_fn.argtypes = [c_ulonglong, c_int]

            # _OperationRegInfo
            MLSL._operation_reg_info_set_name_fn = MLSL.__dll.mlsl_operation_reg_info_set_name
            MLSL._operation_reg_info_set_name_fn.argtypes = [c_ulonglong, c_char_p]

            MLSL._operation_reg_info_add_input_fn = MLSL.__dll.mlsl_operation_reg_info_add_input
            MLSL._operation_reg_info_add_input_fn.argtypes = \
                [c_ulonglong, c_size_t, c_size_t, c_int]

            MLSL._operation_reg_info_add_output_fn = MLSL.__dll.mlsl_operation_reg_info_add_output
            MLSL._operation_reg_info_add_output_fn.argtypes = \
                [c_ulonglong, c_size_t, c_size_t, c_int]

            MLSL._operation_reg_info_add_parameter_set_fn = \
                MLSL.__dll.mlsl_operation_reg_info_add_parameter_set
            MLSL._operation_reg_info_add_parameter_set_fn.argtypes = \
                [c_ulonglong, c_size_t, c_size_t, c_int, c_int]

            MLSL._operation_reg_info_validate_fn = MLSL.__dll.mlsl_operation_reg_info_validate
            MLSL._operation_reg_info_validate_fn.argtypes = [c_ulonglong, c_ulonglong]

            # _Operation
            MLSL._operation_set_distribution_fn = MLSL.__dll.mlsl_operation_set_distribution
            MLSL._operation_set_distribution_fn.argtypes = [c_ulonglong, c_ulonglong]

            MLSL._operation_get_distribution_fn = MLSL.__dll.mlsl_operation_get_distribution
            MLSL._operation_get_distribution_fn.argtypes = [c_ulonglong, POINTER(c_ulonglong)]

            MLSL._operation_get_session_fn = MLSL.__dll.mlsl_operation_get_session
            MLSL._operation_get_session_fn.argtypes = [c_ulonglong, POINTER(c_ulonglong)]

            MLSL._operation_get_session_fn = MLSL.__dll.mlsl_operation_get_session
            MLSL._operation_get_session_fn.argtypes = [c_ulonglong, POINTER(c_ulonglong)]

            MLSL._operation_get_op_type_fn = MLSL.__dll.mlsl_operation_get_op_type
            MLSL._operation_get_op_type_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._operation_set_prev_fn = MLSL.__dll.mlsl_operation_set_prev
            MLSL._operation_set_prev_fn.argtypes = [c_ulonglong, c_ulonglong, c_size_t, c_size_t]

            MLSL._operation_set_next_fn = MLSL.__dll.mlsl_operation_set_next
            MLSL._operation_set_next_fn.argtypes = [c_ulonglong, c_ulonglong, c_size_t, c_size_t]

            MLSL._operation_get_name_fn = MLSL.__dll.mlsl_operation_get_name
            MLSL._operation_get_name_fn.argtypes = [c_ulonglong, POINTER(c_char_p)]

            MLSL._operation_get_global_minibatch_size_fn = \
                MLSL.__dll.mlsl_operation_get_global_minibatch_size
            MLSL._operation_get_global_minibatch_size_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_local_minibatch_size_fn = \
                MLSL.__dll.mlsl_operation_get_local_minibatch_size
            MLSL._operation_get_local_minibatch_size_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_global_minibatch_offset_fn = \
                MLSL.__dll.mlsl_operation_get_global_minibatch_offset
            MLSL._operation_get_global_minibatch_offset_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_input_count_fn = MLSL.__dll.mlsl_operation_get_input_count
            MLSL._operation_get_input_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_input_fn = MLSL.__dll.mlsl_operation_get_input
            MLSL._operation_get_input_fn.argtypes = [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._operation_get_output_count_fn = MLSL.__dll.mlsl_operation_get_output_count
            MLSL._operation_get_output_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_output_fn = MLSL.__dll.mlsl_operation_get_output
            MLSL._operation_get_output_fn.argtypes = [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._operation_has_parameter_sets_fn = MLSL.__dll.mlsl_operation_has_parameter_sets
            MLSL._operation_has_parameter_sets_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._operation_get_parameter_set_count_fn = \
                MLSL.__dll.mlsl_operation_get_parameter_set_count
            MLSL._operation_get_parameter_set_count_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._operation_get_parameter_set_fn = \
                MLSL.__dll.mlsl_operation_get_parameter_set
            MLSL._operation_get_parameter_set_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            # _Statistics
            MLSL._statistics_start_fn = MLSL.__dll.mlsl_statistics_start
            MLSL._statistics_start_fn.argtypes = [c_ulonglong]

            MLSL._statistics_stop_fn = MLSL.__dll.mlsl_statistics_stop
            MLSL._statistics_stop_fn.argtypes = [c_ulonglong]

            MLSL._statistics_reset_fn = MLSL.__dll.mlsl_statistics_reset
            MLSL._statistics_reset_fn.argtypes = [c_ulonglong]

            MLSL._statistics_print_fn = MLSL.__dll.mlsl_statistics_print
            MLSL._statistics_print_fn.argtypes = [c_ulonglong]

            MLSL._statistics_is_started_fn = MLSL.__dll.mlsl_statistics_is_started
            MLSL._statistics_is_started_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._statistics_is_enabled_fn = MLSL.__dll.mlsl_statistics_is_enabled
            MLSL._statistics_is_enabled_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._statistics_get_isolation_comm_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_isolation_comm_cycles
            MLSL._statistics_get_isolation_comm_cycles_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._statistics_get_comm_size_fn = \
                MLSL.__dll.mlsl_statistics_get_comm_size
            MLSL._statistics_get_comm_size_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_size_t)]

            MLSL._statistics_get_comm_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_comm_cycles
            MLSL._statistics_get_comm_cycles_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._statistics_get_compute_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_compute_cycles
            MLSL._statistics_get_compute_cycles_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._statistics_get_total_isolation_comm_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_total_isolation_comm_cycles
            MLSL._statistics_get_total_isolation_comm_cycles_fn.argtypes = \
                [c_ulonglong, POINTER(c_ulonglong)]

            MLSL._statistics_get_total_comm_size_fn = \
                MLSL.__dll.mlsl_statistics_get_total_comm_size
            MLSL._statistics_get_total_comm_size_fn.argtypes = \
                [c_ulonglong, POINTER(c_size_t)]

            MLSL._statistics_get_total_comm_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_total_comm_cycles
            MLSL._statistics_get_total_comm_cycles_fn.argtypes = \
                [c_ulonglong, POINTER(c_ulonglong)]

            MLSL._statistics_get_total_compute_cycles_fn = \
                MLSL.__dll.mlsl_statistics_get_total_compute_cycles
            MLSL._statistics_get_total_compute_cycles_fn.argtypes = \
                [c_ulonglong, POINTER(c_ulonglong)]

            # Session
            MLSL._session_set_global_minibatch_size_fn = \
                MLSL.__dll.mlsl_session_set_global_minibatch_size
            MLSL._session_set_global_minibatch_size_fn.argtypes = [c_ulonglong, c_size_t]

            MLSL._session_get_global_minibatch_size_fn = \
                MLSL.__dll.mlsl_session_get_global_minibatch_size
            MLSL._session_get_global_minibatch_size_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._session_get_phase_type_fn = MLSL.__dll.mlsl_session_get_phase_type
            MLSL._session_get_phase_type_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._session_create_operation_reg_info_fn = \
                MLSL.__dll.mlsl_session_create_operation_reg_info
            MLSL._session_create_operation_reg_info_fn.argtypes = \
                [c_ulonglong, c_int, POINTER(c_ulonglong)]

            MLSL._session_delete_operation_reg_info_fn = \
                MLSL.__dll.mlsl_session_delete_operation_reg_info
            MLSL._session_delete_operation_reg_info_fn.argtypes = [c_ulonglong, c_ulonglong]

            MLSL._session_add_operation_with_distribution_fn =  \
                MLSL.__dll.mlsl_session_add_operation_with_distribution
            MLSL._session_add_operation_with_distribution_fn.argtypes = \
                [c_ulonglong, c_ulonglong, c_ulonglong, POINTER(c_size_t)]

            MLSL._session_add_operation_fn = MLSL.__dll.mlsl_session_add_operation
            MLSL._session_add_operation_fn.argtypes = \
                [c_ulonglong, c_ulonglong, POINTER(c_size_t)]

            MLSL._session_remove_operations_fn = MLSL.__dll.mlsl_session_remove_operations
            MLSL._session_remove_operations_fn.argtypes = [c_ulonglong]

            MLSL._session_get_operation_count_fn = MLSL.__dll.mlsl_session_get_operation_count
            MLSL._session_get_operation_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._session_get_operation_fn = MLSL.__dll.mlsl_session_get_operation
            MLSL._session_get_operation_fn.argtypes = \
                [c_ulonglong, c_size_t, POINTER(c_ulonglong)]

            MLSL._session_commit_fn = MLSL.__dll.mlsl_session_commit
            MLSL._session_commit_fn.argtypes = [c_ulonglong]

            MLSL._session_get_stats_fn = MLSL.__dll.mlsl_session_get_stats
            MLSL._session_get_stats_fn.argtypes = [c_ulonglong, POINTER(c_ulonglong)]

            # Environment
            MLSL._environment_get_version_fn = MLSL.__dll.mlsl_environment_get_version
            MLSL._environment_get_version_fn.argtypes = [POINTER(c_int)]

            MLSL._environment_configure_fn = MLSL.__dll.mlsl_environment_configure
            MLSL._environment_configure_fn.argtypes = [c_ulonglong, c_char_p]

            MLSL._environment_init_fn = MLSL.__dll.mlsl_environment_init
            MLSL._environment_init_fn.argtypes = \
                [c_ulonglong, POINTER(c_int), POINTER(POINTER(c_char_p))]

            MLSL._environment_finalize_fn = MLSL.__dll.mlsl_environment_finalize
            MLSL._environment_finalize_fn.argtypes = [c_ulonglong]

            MLSL._environment_is_initialized_fn = MLSL.__dll.mlsl_environment_is_initialized
            MLSL._environment_is_initialized_fn.argtypes = [c_ulonglong, POINTER(c_int)]

            MLSL._environment_get_process_idx_fn = MLSL.__dll.mlsl_environment_get_process_idx
            MLSL._environment_get_process_idx_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._environment_get_process_count_fn = MLSL.__dll.mlsl_environment_get_process_count
            MLSL._environment_get_process_count_fn.argtypes = [c_ulonglong, POINTER(c_size_t)]

            MLSL._environment_create_session_fn = MLSL.__dll.mlsl_environment_create_session
            MLSL._environment_create_session_fn.argtypes = \
                [c_ulonglong, c_int, POINTER(c_ulonglong)]

            MLSL._environment_delete_session_fn = MLSL.__dll.mlsl_environment_delete_session
            MLSL._environment_delete_session_fn.argtypes = [c_ulonglong, c_ulonglong]

            MLSL._environment_create_distribution_fn = \
                MLSL.__dll.mlsl_environment_create_distribution
            MLSL._environment_create_distribution_fn.argtypes = \
                [c_ulonglong, c_size_t, c_size_t, POINTER(c_ulonglong)]

            MLSL._environment_delete_distribution_fn = \
                MLSL.__dll.mlsl_environment_delete_distribution
            MLSL._environment_delete_distribution_fn.argtypes = [c_ulonglong, c_ulonglong]

            MLSL._environment_wait_fn = MLSL.__dll.mlsl_environment_wait
            MLSL._environment_wait_fn.argtypes = [c_ulonglong, c_ulonglong]

            MLSL._environment_test_fn = MLSL.__dll.mlsl_environment_test
            MLSL._environment_test_fn.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_int)]

            MLSL._environment_alloc_fn = MLSL.__dll.mlsl_environment_alloc
            MLSL._environment_alloc_fn.argtypes = \
                [c_ulonglong, c_size_t, c_size_t, POINTER(c_void_p)]

            MLSL._environment_free_fn = MLSL.__dll.mlsl_environment_free
            MLSL._environment_free_fn.argtypes = [c_ulonglong, c_void_p]

            MLSL.__is_initialized = True

        elif MLSL.__is_loaded is False:
            logging.error("Intel(R) MLSL library isn't loaded, exiting ...")
            sys.exit(1)

        else:
            pass


def close():
    if mlsl_module._mlsl_obj is not None:
        logging.debug("Intel(R) MLSL: Cleaning things up")
        if mlsl_module._ref_count != 1:
            raise RuntimeError("Unexpected reference count for Intel(R) MLSL object: %d",
                               mlsl_module._ref_count)
        mlsl_module._mlsl_obj.finalize()
        mlsl_module._mlsl_obj = None

if os.getenv("MLSL_ALLOW_REINIT") == "1":
    mlsl_module._mlsl_obj = MLSL()
    mlsl_module._mlsl_obj.init()
