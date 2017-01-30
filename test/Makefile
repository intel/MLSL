# GNU Makefile that builds and runs example.
#
# Copyright (C) 2016 Intel Corporation.
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

all: release

INCLUDES    = -I$(MLSL_ROOT)/intel64/include
LDFLAGS     = -L$(MLSL_ROOT)/intel64/lib -lmlsl -lrt
CXX         = icpc
CFLAGS      = -O3
tests       = mlsl_test
CXX_COMPILE = $(CXX) $(CFLAGS) $(INCLUDES)

# FIXME: remove that after moving to the new MLSL API
ifneq ($(CXX), icpc)
  CFLAGS += -fpermissive
endif

.SUFFIXES:
.SUFFIXES: .cpp

.cpp:
	$(CXX_COMPILE) -o $* $< $(LDFLAGS)

release: $(tests)

test: all
	@echo "Running Data parallelism mode..."
	@if [ "x$(VERBOSE)" == "x1" ] ; then \
	  echo "mpiexec.hydra -n 4 ./$(tests) 1 > Data_parallel.log 2>&1" ; \
	fi
	@mpiexec.hydra -n 4 ./$(tests) 1 > Data_parallel.log 2>&1
	@echo "Result in Data_parallel.log"
	
	@echo "Running Model parallelism mode..."
	@if [ "x$(VERBOSE)" == "x1" ] ; then \
	  echo "mpiexec.hydra -n 4 ./$(tests) 4 > Model_parallel.log 2>&1" ; \
	fi
	@mpiexec.hydra -n 4 ./$(tests) 4 > Model_parallel.log 2>&1
	@echo "Result in Model_parallel.log"
	
	@echo "Running Hybrid parallelism (N/2) x 2 mode..."
	@if [ "x$(VERBOSE)" == "x1" ] ; then \
	  echo "mpiexec.hydra -n 4 ./$(tests) 2 > Hybrid_parallel.log 2>&1" ; \
	fi
	@mpiexec.hydra -n 4 ./$(tests) 2 > Hybrid_parallel.log 2>&1
	@echo "Result in Hybrid_parallel.log"

clean:
	-rm -f $(tests).o $(tests) *.log
