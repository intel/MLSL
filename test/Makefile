
#
# Copyright (C) 2017 Intel Corporation.
# 
# The Intel(R) Machine Learning Scaling Library ("Software") is furnished under
# license and may only be used or copied in accordance with the terms of that
# license. No license, express or implied, by estoppel or otherwise, to any
# intellectual property rights is granted by this document. The Software is
# subject to change without notice, and should not be construed as a commitment
# by Intel Corporation to market, license, sell or support any product or
# technology. Unless otherwise provided for in the license under which this
# Software is provided, the Software is provided AS IS, with no warranties of any
# kind, express or implied. Except as expressly permitted by the Software license,
# neither Intel Corporation nor its suppliers assumes any responsibility or
# liability for any errors or inaccuracies that may appear herein. Except as
# expressly permitted by the Software license, no part of the Software may be
# reproduced, stored in a retrieval system, transmitted in any form, or
# distributed by any means without the express written consent of
# Intel Corporation.
#
ENABLE_DEBUG ?= 1

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
			done ; \
		done ; \
	done ;
	if [ $(PATH_TO_QUANT_LIB) ] ; then \
		echo "running mlsl_test with quantization ..." ; \
		echo "group_count = 1, dist_update = 0, user_buf = 0, stats = 0" >> output_with_quant.log 2>&1 ; \
		mpiexec.hydra -n 4 -ppn 1 -l ./$(tests) 1 0 0 0 $(PATH_TO_QUANT_LIB) >> output_with_quant.log 2>&1 ; \
	fi ;
	echo "running mlsl_test.py ..." ; \
	for group_count in 1 2 4 ; do \
		for usr_buf in 0 ; do \
			for dist_upd in 1 0 ; do \
				echo "group_count = $$group_count, dist_update = $$dist_upd, user_buf = $$usr_buf, stats = 0" >> output_py.log 2>&1 ; \
				mpiexec.hydra -n 4 -ppn 1 -l python ./$(tests).py $$group_count $$dist_upd $$usr_buf 0 >> output_py.log 2>&1 ; \
			done ; \
		done ; \
	done ;

clean:
	-rm -f $(tests).o $(tests) *.log
