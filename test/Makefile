
#
# Copyright 2018 Intel Corporation.
# 
# This software and the related documents are Intel copyrighted materials, and your use of
# them is governed by the express license under which they were provided to you (License).
# Unless the License provides otherwise, you may not use, modify, copy, publish, distribute,
# disclose or transmit this software or the related documents without Intel's prior written
# permission.
# 
# This software and the related documents are provided as is, with no express or implied
# warranties, other than those that are expressly stated in the License.
#
ENABLE_DEBUG ?= 1
SHELL = bash
INCLUDES = -I$(MLSL_ROOT)/intel64/include
LDFLAGS  = -L$(MLSL_ROOT)/intel64/lib -lmlsl -lrt
CXX     ?= g++
tests    = mlsl_test

ifeq ($(ENABLE_DEBUG),1)
  CFLAGS += -O0 -g
else
  CFLAGS += -O2
endif

CFLAGS  += $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

CXX_COMPILE = $(CXX) $(CFLAGS) $(INCLUDES)

.SUFFIXES:
.SUFFIXES: .cpp

.cpp:
	$(CXX_COMPILE) -o $* $< $(LDFLAGS)

all: $(tests)

run: all
	-rm -f *.log
	echo "running mlsl_test ..." ; \
	for group_count in 1 2 4 ; do \
		for usr_buf in 1 0 ; do \
			for dist_upd in 1 0 ; do \
				echo "group_count = $$group_count, dist_update = $$dist_upd, user_buf = $$usr_buf, stats = 0" >> output.log 2>&1 ; \
				mpiexec.hydra -n 4 -ppn 1 -l ./$(tests) $$group_count $$dist_upd $$usr_buf 0 >> output.log 2>&1 ; \
				if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
					echo "Run FAILED." ; \
				fi ; \
			done ; \
		done ; \
	done ;
	if [ $(PATH_TO_QUANT_LIB) ] ; then \
		echo "running mlsl_test with quantization ..." ; \
		echo "group_count = 1, dist_update = 0, user_buf = 0, stats = 0" >> output_with_quant.log 2>&1 ; \
		mpiexec.hydra -n 4 -ppn 1 -l ./$(tests) 1 0 0 0 $(PATH_TO_QUANT_LIB) >> output_with_quant.log 2>&1 ; \
		if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
			echo "Run FAILED." ; \
		fi ; \
	fi ;
	echo "running mlsl_test.py ..." ; \
	for group_count in 1 2 4 ; do \
		for usr_buf in 0 ; do \
			for dist_upd in 1 0 ; do \
				echo "group_count = $$group_count, dist_update = $$dist_upd, user_buf = $$usr_buf, stats = 0" >> output_py.log 2>&1 ; \
				mpiexec.hydra -n 4 -ppn 1 -l python ./$(tests).py $$group_count $$dist_upd $$usr_buf 0 >> output_py.log 2>&1 ; \
				if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
					echo "Run FAILED." ; \
				fi ; \
			done ; \
		done ; \
	done ;

clean:
	-rm -f $(tests).o $(tests) *.log
