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
tests    = mlsl_example

ifeq ($(ENABLE_DEBUG),1)
  CFLAGS += -O0 -g
else
  CFLAGS += -O2
endif

CXX_COMPILE = $(CXX) $(CFLAGS) $(INCLUDES)

.SUFFIXES:
.SUFFIXES: .cpp

.cpp:
	$(CXX_COMPILE) -o $* $< $(LDFLAGS)

all: $(tests)

run: all
	-rm -f *.log
	echo "running mlsl_example ..." ; \
	for group_count in 1 2 4 ; do \
		echo "mpiexec.hydra -n 4 -ppn 1 -l ./$(tests) $$group_count" >> output.log 2>&1 ; \
		mpiexec.hydra -n 4 -ppn 1 -l ./$(tests) $$group_count >> output.log 2>&1 ; \
		if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
			echo "Run FAILED." ; \
		fi ; \
	done;

clean:
	-rm -f $(tests).o $(tests) *.log
