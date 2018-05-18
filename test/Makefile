
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
