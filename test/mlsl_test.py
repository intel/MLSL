#!/usr/bin/python
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

# Intel(R) MLSL library API usage example and correctness check test

#from builtins import range
from collections import namedtuple
import mlsl
import numpy as np
from math import fabs
import sys
import ctypes

mlsl_obj = mlsl.MLSL()

dtype_size = 8
np_type = "float32" if dtype_size == 4 else "float64"
mlsl_dtype = mlsl.DataType.FLOAT if dtype_size == 4 else mlsl.DataType.DOUBLE
cacheline_size = 64
fail_counter_max = 5

global_minibatch_size = 16
layer_count = 2
epoch_count = 2
minibatch_per_epoch = 1

process_idx = None
process_count = 0

group_count = 1
use_dist_update = False  # data parallelism's feature
use_user_buf = False
use_test = False

layer_type_conv_mimo = 0
layer_type_conv_flat = 1
layer_type_fc = 2

layers = [None] * layer_count
operations = [None] * layer_count

LayerParam = namedtuple("LayerParam", "layer_idx type ifm ofm ifm_w ifm_h ofm_w ofm_h k_w k_h")
layer_params = [None] * layer_count


class Layer(object):
    def __init__(self, layer_idx, op, prev_layer):
        self.layer_idx = layer_idx
        self.op = op
        self.input_act_arr = None
        self.output_act_arr = None
        self.input_act_grad_arr = None
        self.output_act_grad_arr = None
        self.param_arr = None
        self.param_grad_arr = None
        self.param_inc_arr = None
        self.is_backward_unpack_called = False
        assert (self.op.get_input(0) is not None), "input activation is null"
        assert (self.op.get_parameter_set(0) is not None), "parameter is null"

        in_act_size = 0
        prev_out_act_size = 0
        if prev_layer is None:
            prev_out_act_size = 0
        else:
            assert (self.op.get_output(0) is not None), "output activation is null"
            output_act = prev_layer.op.get_output(0)
            prev_out_act_size = output_act.get_local_fm_count() \
                * prev_layer.op.get_local_minibatch_size() \
                * output_act.get_fm_size() * dtype_size

        input_act = self.op.get_input(0)
        in_act_size = input_act.get_local_fm_count() \
            * self.op.get_local_minibatch_size() \
            * input_act.get_fm_size() * dtype_size

        if prev_out_act_size > in_act_size:
            in_act_size = prev_out_act_size

        self.input_act_arr = create_array(in_act_size)
        self.input_act_grad_arr = create_array(in_act_size)

        if prev_layer is not None:
            prev_layer.output_act_arr = self.input_act_arr
            prev_layer.output_act_grad_arr = self.input_act_grad_arr
            self.op.set_prev(prev_layer.op, 0, 0)

        self.param_arr_count = self.op.get_parameter_set(0).get_local_kernel_count() \
            * self.op.get_parameter_set(0).get_kernel_size()
        self.param_arr_size = self.param_arr_count * dtype_size

        param_arr_inc_size = self.op.get_parameter_set(0).get_owned_kernel_count() \
            * self.op.get_parameter_set(0).get_kernel_size() * dtype_size

        if use_user_buf is True:
            assert False, "unsupported case"
        else:
            self.param_arr = create_array(self.param_arr_size)
            self.param_grad_arr = create_array(self.param_arr_size)
            self.param_inc_arr = create_array(param_arr_inc_size)

        assert ((self.input_act_arr is not None) and (self.input_act_grad_arr is not None)
                and (self.param_arr is not None) and (self.param_grad_arr is not None)
                and (self.param_inc_arr is not None)), "error while arrays creating"

        for idx in range(0, self.param_arr.size):
            self.param_arr[idx] = idx

    def destroy(self):
        mlsl_obj.free(np.ctypeslib.as_ctypes(self.input_act_arr))
        mlsl_obj.free(np.ctypeslib.as_ctypes(self.input_act_grad_arr))

        if use_user_buf is True:
            assert False, "unsupported case"
        else:
            mlsl_obj.free(np.ctypeslib.as_ctypes(self.param_arr))
            mlsl_obj.free(np.ctypeslib.as_ctypes(self.param_grad_arr))
            mlsl_obj.free(np.ctypeslib.as_ctypes(self.param_inc_arr))

    def get_param_arr_count(self):
        return self.param_arr_count

    def get_param_arr(self):
        return np.ctypeslib.as_ctypes(self.param_arr)

    def pack_buffer(self, act, comm_arr, local_arr):
        local_fm_count = act.get_local_fm_count()
        for block_idx in range(0, act.get_pack_block_count()):
            block_info = act.get_pack_block(block_idx)
            mb_count = block_info.get_mb_count()
            mb_offset = block_info.get_mb_offset()
            fm_count = block_info.get_fm_count()
            fm_offset = block_info.get_fm_offset()
            fm_size = block_info.get_fm_size()
            src = local_arr
            dst = comm_arr[block_info.get_buf_offset():]
            for mb_idx in range(0, mb_count):
                for fm_idx in range(0, fm_count):
                    for space_idx in range(0, fm_size):
                        dst_idx = mb_idx * fm_count * fm_size + fm_idx * fm_size + space_idx
                        src_idx = (mb_idx + mb_offset) * local_fm_count * fm_size \
                            + (fm_idx + fm_offset) * fm_size \
                            + space_idx
                        dst[dst_idx] = src[src_idx]

    def unpack_buffer(self, act, comm_arr, local_arr):
        local_fm_count = act.get_local_fm_count()
        for block_idx in range(0, act.get_unpack_block_count()):
            block_info = act.get_unpack_block(block_idx)
            mb_count = block_info.get_mb_count()
            mb_offset = block_info.get_mb_offset()
            fm_count = block_info.get_fm_count()
            fm_offset = block_info.get_fm_offset()
            fm_size = block_info.get_fm_size()
            src = comm_arr[block_info.get_buf_offset():]
            dst = local_arr
            for mb_idx in range(0, mb_count):
                for fm_idx in range(0, fm_count):
                    for space_idx in range(0, fm_size):
                        dst_idx = (mb_idx + mb_offset) * local_fm_count * fm_size \
                            + (fm_idx + fm_offset) * fm_size \
                            + space_idx
                        src_idx = mb_idx * fm_count * fm_size + fm_idx * fm_size + space_idx
                        dst[dst_idx] = src[src_idx]

    def forward_compute(self):
        if self.layer_idx == 0:
            # Write to output activation
            assert (self.op.get_output(0) is not None), "output activation is null"
            output_act = self.op.get_output(0)
            out_size = output_act.get_local_fm_count() * self.op.get_local_minibatch_size() \
                * output_act.get_fm_size()
            for idx in range(0, out_size):
                self.output_act_arr[idx] = idx
        elif self.layer_idx == 1:
            # Check for input activation
            assert (self.op.get_input(0) is not None), "input activation is null"
            fm_local_count = self.op.get_input(0).get_local_fm_count()
            mb_local_len = self.op.get_local_minibatch_size()
            fm_size = self.op.get_input(0).get_fm_size()
            fm_offset = self.op.get_input(0).get_global_fm_offset()
            fm_group_size = self.op.get_distribution().get_process_count(mlsl.GroupType.MODEL)
            fail_counter = 0

            for mb_idx in range(0, mb_local_len):
                for fm_idx in range(0, fm_local_count):
                    for space_idx in range(0, fm_size):
                        expected = fm_group_size * (mb_idx * fm_local_count
                                                    * fm_size * fm_group_size
                                                    + (fm_offset + fm_idx) * fm_size
                                                    + space_idx)
                        idx = mb_idx * fm_local_count * fm_size + fm_idx * fm_size + space_idx
                        if fabs(self.input_act_arr[idx] - expected) > 1.e-4:
                            if fail_counter < fail_counter_max:
                                print('[%u] forward_%u: input: idx %u: '
                                      'expected %4.0f - received: %4.0f\n'
                                      % (process_idx, self.layer_idx, idx,
                                         expected, self.input_act_arr[idx]))
                            fail_counter += 1

            if fail_counter > 0:
                print('[%u] forward_%u: input activation test FAILED mismatch count = %u\n'
                      % (process_idx, self.layer_idx, fail_counter))
                assert False, "exit"
            else:
                print('[%u] forward_%u: input activation test PASSED\n'
                      % (process_idx, self.layer_idx))

        # Now check ParameterSet
        assert (self.op.get_parameter_set(0) is not None), "parameter is null"
        param_size = self.op.get_parameter_set(0).get_local_kernel_count() \
            * self.op.get_parameter_set(0).get_kernel_size()
        fail_counter = 0
        for idx in range(0, param_size):
            if fabs(self.param_arr[idx] - idx) > 1.e-4:
                if fail_counter < fail_counter_max:
                    print('[%u] forward_%u: parameter idx %u: expected %4.0f - received: %4.0f\n'
                          % (process_idx, self.layer_idx, idx,
                             idx, self.param_arr[idx]))
                fail_counter += 1

        if fail_counter > 0:
            print('[%u] forward_%u: parameter test FAILED mismatch count = %u\n'
                  % (process_idx, self.layer_idx, fail_counter))
            assert False, "exit"
        else:
            print('[%u] forward_%u: parameter test PASSED\n' % (process_idx, self.layer_idx))

    def backward_compute1(self):
        if self.layer_idx == 0:
            # Check for inputs
            assert (self.op.get_output(0) is not None), "output activation is null"
            act_size = self.op.get_output(0).get_local_fm_count() \
                * self.op.get_local_minibatch_size() \
                * self.op.get_output(0).get_fm_size()
            fail_counter = 0
            for idx in range(0, act_size):
                if fabs(self.output_act_grad_arr[idx] - idx) > 1.e-4:
                    if fail_counter < fail_counter_max:
                        print('[%u] backward_%u: output activation gradient: idx %u: '
                              'expected %4.0f - received: %4.0f\n'
                              % (process_idx, self.layer_idx, idx,
                                 idx, self.output_act_grad_arr[idx]))
                    fail_counter += 1

            if fail_counter > 0:
                print('[%u] backward_%u: output activation gradient '
                      'test FAILED mismatch count = %u\n'
                      % (process_idx, self.layer_idx, fail_counter))
                assert False, "exit"
            else:
                print('[%u] backward_%u: output activation gradient test PASSED\n'
                      % (process_idx, self.layer_idx))
        elif self.layer_idx == 1:
            # Write to output
            assert (self.op.get_input(0) is not None), "input activation is null"
            fm_local_count = self.op.get_input(0).get_local_fm_count()
            mb_local_len = self.op.get_local_minibatch_size()
            fm_size = self.op.get_input(0).get_fm_size()
            act_offset = self.op.get_input(0).get_global_fm_offset()
            group_size = self.op.get_distribution().get_process_count(mlsl.GroupType.MODEL)
            for mb_idx in range(0, mb_local_len):
                for fm_idx in range(0, fm_local_count):
                    for space_idx in range(0, fm_size):
                        idx = mb_idx * fm_local_count * fm_size + fm_idx * fm_size + space_idx
                        self.input_act_grad_arr[idx] = mb_idx * fm_local_count \
                            * fm_size * group_size \
                            + (act_offset + fm_idx) * fm_size \
                            + space_idx

    def backward_compute2(self):
        assert (self.op.get_parameter_set(0) is not None), "parameter is null"
        param_size = self.op.get_parameter_set(0).get_local_kernel_count() \
            * self.op.get_parameter_set(0).get_kernel_size()
        for idx in range(0, param_size):
            self.param_grad_arr[idx] = idx

    def update_compute(self, param_grad, param_inc, owned_param, owned_size):
        assert (self.op.get_parameter_set(0) is not None), "parameter is null"
        mb_group_size = self.op.get_distribution().get_process_count(mlsl.GroupType.DATA)
        owned_offset = self.op.get_parameter_set(0).get_owned_kernel_offset() \
            * self.op.get_parameter_set(0).get_kernel_size()
        fail_counter = 0
        for idx in range(0, owned_size):
            expected = mb_group_size * (owned_offset + idx)
            if fabs(param_grad[idx] - expected) > 1.e-4:
                fail_counter += 1
            owned_param[idx] = owned_offset + idx

        if fail_counter > 0:
            print('[%u] update_%u: parameter gradient test FAILED mismatch count = %u\n'
                  % (process_idx, self.layer_idx, fail_counter))
            assert False, "exit"
        else:
            print('[%u] update_%u: parameter gradient test PASSED\n'
                  % (process_idx, self.layer_idx))

    # Recv parameter increments (in case of distributed update) and input activations,
    # and send output activations
    def forward(self):
        assert (self.op.get_input(0) is not None), "input activation is null"
        assert (self.op.get_output(0) is not None), "otput activation is null"

        act = self.op.get_input(0)
        comm_buf = act.wait_comm()
        comm_arr = create_array(act.get_comm_buf_size(), comm_buf)
        self.unpack_buffer(act, comm_arr, self.input_act_arr)
        if self.op.has_parameter_sets() is True:
            assert (self.op.get_parameter_set(0) is not None), "parameter is null"
            self.op.get_parameter_set(0).wait_increment_comm()

        self.forward_compute()

        act = self.op.get_output(0)
        output_act_comm_buf = act.get_comm_buf()
        output_act_comm_arr = create_array(act.get_comm_buf_size(), output_act_comm_buf)
        self.pack_buffer(act, output_act_comm_arr, self.output_act_arr)
        act.start_comm(output_act_comm_buf)
        self.is_backward_unpack_called = False

    # Calculate gradient wrt input activation and send it
    def backward1(self):
        assert (self.op.get_input(0) is not None), "input activation is null"

        if self.is_backward_unpack_called is False:
            assert (self.op.get_output(0) is not None), "output activation is null"
            act = self.op.get_output(0)
            comm_buf = act.wait_comm()
            comm_arr = create_array(act.get_comm_buf_size(), comm_buf)
            self.unpack_buffer(act, comm_arr, self.output_act_grad_arr)
            self.is_backward_unpack_called = True

        self.backward_compute1()

        act = self.op.get_input(0)
        input_act_comm_buf = act.get_comm_buf()
        input_act_comm_arr = create_array(act.get_comm_buf_size(), input_act_comm_buf)
        self.pack_buffer(act, input_act_comm_arr, self.input_act_grad_arr)
        act.start_comm(input_act_comm_buf)

    # Calculate gradient wrt parameters and send it
    def backward2(self):
        if self.is_backward_unpack_called is False:
            assert (self.op.get_output(0) is not None), "output activation is null"
            act = self.op.get_output(0)
            comm_buf = act.wait_comm()
            comm_arr = create_array(act.get_comm_buf_size(), comm_buf)
            self.unpack_buffer(act, comm_arr, self.output_act_grad_arr)
            self.is_backward_unpack_called = True

        self.backward_compute2()

        if self.op.has_parameter_sets() is True:
            assert (self.op.get_parameter_set(0) is not None), "parameter is null"
            self.op.get_parameter_set(0).start_gradient_comm(array_to_pointer(self.param_grad_arr))

    # Recv gradient wrt parameters and update parameters/send parameter increments
    # (in case of distributed update)
    def update(self):
        if self.op.has_parameter_sets() is True:
            assert (self.op.get_parameter_set(0) is not None), "parameter is null"
            comm_buf = None
            if use_test is True:
                is_completed = False
                while is_completed is False:
                    comm_buf, is_completed = self.op.get_parameter_set(0).test_gradient_comm()
            else:
                comm_buf = self.op.get_parameter_set(0).wait_gradient_comm()

            if process_count == 1:
                assert (comm_buf is None), "comm_buf should be none for single node"

            owned_param_arr = \
                self.param_arr[(self.op.get_parameter_set(0).get_owned_kernel_offset()
                                * self.op.get_parameter_set(0).get_kernel_size()):]
            self.update_compute(self.param_grad_arr if comm_buf is None
                                else create_array(self.param_arr_size, comm_buf),
                                self.param_inc_arr,
                                owned_param_arr,
                                self.op.get_parameter_set(0).get_owned_kernel_count()
                                * self.op.get_parameter_set(0).get_kernel_size())
            self.op.get_parameter_set(0).start_increment_comm(array_to_pointer(self.param_arr))


def create_array(size, buf=None):
    # print('create_array: mlsl_dtype {}, size {}, dtype_size {}, size/dtype_size {}'\
    #       .format(mlsl_dtype, size, dtype_size, (size/dtype_size)))
    if size == 0:
        return np.empty(0, dtype=np_type)
    if buf is None:
        buf = mlsl_obj.alloc(size, cacheline_size)
    if dtype_size == 4:
        buf_pointer = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float * int(size / dtype_size)))
    elif dtype_size == 8:
        buf_pointer = ctypes.cast(buf, ctypes.POINTER(ctypes.c_double * int(size / dtype_size)))
    else:
        assert False, "unsupported case"
    array = np.frombuffer(buf_pointer.contents, dtype=np_type)
    return array


def array_to_pointer(array):
    return ctypes.cast(np.ctypeslib.as_ctypes(array), ctypes.c_void_p)


# Layer initialization
def create_layer(session, layer_type, layer_params, distribution, prev_layer):
    assert (layer_type == layer_type_conv_mimo) \
        or (layer_type == layer_type_conv_flat) \
        or (layer_type == layer_type_fc), \
        "incorrect layer type"

    layer_idx = layer_params.layer_idx
    reg_info = session.create_operation_reg_info(mlsl.OperationType.CC)
    reg_info.set_name(("layer_" + str(layer_idx)).encode('utf-8'))
    reg_info.add_input(layer_params.ifm, layer_params.ifm_w * layer_params.ifm_h, mlsl_dtype)
    reg_info.add_output(layer_params.ofm, layer_params.ofm_w * layer_params.ofm_h, mlsl_dtype)
    reg_info.add_parameter_set(layer_params.ifm * layer_params.ofm,
                               layer_params.k_w * layer_params.k_h,
                               mlsl_dtype,
                               use_dist_update)

    op_idx = session.add_operation_with_distribution(reg_info, distribution)
    session.delete_operation_reg_info(reg_info)
    op = session.get_operation(op_idx)
    operations[layer_idx] = op
    layer = Layer(layer_idx, op, prev_layer)
    return layer


def main():
    argc = len(sys.argv)
    if argc < 2:
        print('specify parameters: mlsl_test.py GROUP_COUNT [DIST_UPDATE] [USER_BUF] [USE_TEST]')
        sys.exit(0)

    global group_count
    global use_dist_update
    global use_user_buf
    global use_test
    global process_idx
    global process_count

    mlsl_obj.init()
    session = mlsl_obj.create_session(mlsl.PhaseType.TRAIN)
    session.set_global_minibatch_size(global_minibatch_size)
    process_count = mlsl_obj.get_process_count()

    if argc > 1:
        group_count = int(sys.argv[1])

    if argc > 2:
        use_dist_update = (True if (int(sys.argv[2]) != 0) else False)

    if argc > 3:
        use_user_buf = (True if (int(sys.argv[3]) != 0) else False)

    if argc > 4:
        use_test = (True if (int(sys.argv[4]) != 0) else False)

    if group_count < 1:
        group_count = 1

    if group_count > process_count:
        group_count = process_count

    process_idx = mlsl_obj.get_process_idx()
    if process_idx == 0:
        print('process_count = %u, distribution = %u x %u (data_parts x model_parts), '
              'dist_update %d, user_buf %d, use_test %d\n'
              % (process_count, int(process_count / group_count),
                 group_count, use_dist_update, use_user_buf, use_test))

    # Correctness test assumes both the layers use same distribution
    distribution = mlsl_obj.create_distribution(int(process_count / group_count), group_count)

    # Init all the layers
    for layer_idx in range(0, layer_count):
        # Set layer_params for each layer
        if layer_idx == 0:
            layer_params[layer_idx] = LayerParam(layer_idx=layer_idx, type=layer_type_conv_mimo,
                                                 ifm=128, ofm=256,
                                                 ifm_w=12, ifm_h=12,
                                                 ofm_w=12, ofm_h=12,
                                                 k_w=3, k_h=3)
        elif layer_idx == 1:
            layer_params[layer_idx] = LayerParam(layer_idx=layer_idx, type=layer_type_conv_mimo,
                                                 ifm=256, ofm=256,
                                                 ifm_w=12, ifm_h=12,
                                                 ofm_w=12, ofm_h=12,
                                                 k_w=3, k_h=3)
        layers[layer_idx] = create_layer(session, layer_params[layer_idx].type,
                                         layer_params[layer_idx], distribution,
                                         None if layer_idx == 0 else layers[layer_idx - 1])
        req = distribution.bcast(layers[layer_idx].get_param_arr(),
                                 layers[layer_idx].get_param_arr_count(),
                                 mlsl_dtype, 0, mlsl.GroupType.GLOBAL)
        mlsl_obj.wait(req)

    session.commit()

    stats = session.get_stats()
    stats.start()

    for epoch_idx in range(0, epoch_count):
        for mb_idx in range(0, minibatch_per_epoch):
            for layer_idx in range(0, layer_count):
                layers[layer_idx].forward()

            for layer_idx in range(0, layer_count):
                # Split backward phase on 2 steps to achieve comp/comm overlapping
                layers[layer_count - layer_idx - 1].backward1()
                layers[layer_count - layer_idx - 1].backward2()

            for layer_idx in range(0, layer_count):
                layers[layer_idx].update()

            if stats.is_enabled() is True:
                print('\n ++[%u] total isolation comm cycles: %lu'
                      % (process_idx, stats.get_total_isolation_comm_cycles()))
                print('\n ++[%u] total communication bytes: %u'
                      % (process_idx, stats.get_total_comm_size()))
                print('\n ++[%u] total communication cycles: %lu'
                      % (process_idx, stats.get_total_comm_cycles()))
                print('\n ++[%u] total compute cycles: %lu\n'
                      % (process_idx, stats.get_total_compute_cycles()))

        # Finish ParameterSet comms before ending epoch
        for layer_idx in range(0, layer_count):
            op = operations[layer_idx]
            if op.has_parameter_sets() is True:
                assert (op.get_parameter_set(0) is not None), "parameter is null"
                op.get_parameter_set(0).wait_increment_comm()

    stats.stop()
    stats.dump()

    for layer_idx in range(0, layer_count):
        layers[layer_idx].destroy()

    mlsl_obj.delete_session(session)
    mlsl_obj.delete_distribution(distribution)
    mlsl_obj.finalize()

    print('[%u] exited normally\n' % (process_idx))


main()
