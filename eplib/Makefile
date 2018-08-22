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
USE_SECURITY_FLAGS ?= 1
ENABLE_DEBUG       ?= 0

CC         ?= gcc
BASE_DIR    = $(shell pwd)
MPIRT_DIR  ?= $(BASE_DIR)/../mpirt
CFLAGS      += -I$(MPIRT_DIR)/include -I$(BASE_DIR)/../quant

QUANT_DIR = $(BASE_DIR)/../quant

ifeq ($(USE_SECURITY_FLAGS),1)
    SECURITY_CXXFLAGS   = -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector
    SECURITY_LDFLAGS    = -z noexecstack -z relro -z now
    CFLAGS              += $(SECURITY_CXXFLAGS)
    LDFLAGS             += $(SECURITY_LDFLAGS)
endif

CFLAGS  += $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

ifeq ($(ENABLE_DEBUG), 1)
  CLIENT_FLAGS = -DENABLE_DEBUG -O0 $(CFLAGS) -g
  SERVER_FLAGS = -DENABLE_DEBUG -O0 $(CFLAGS) -g
else
  CLIENT_FLAGS = -O2 $(CFLAGS)
  SERVER_FLAGS = -O2 $(CFLAGS)
endif

CLIENT_FLAGS        += -Wall -std=gnu99 -I. -fPIC -DENABLE_CLIENT_ONLY -DENABLE_FASTPATH -D_GNU_SOURCE -DENABLE_ASYNC_PROGRESS
SERVER_FLAGS        += -Wall -std=gnu99 -I. -DENABLE_FASTPATH -D_GNU_SOURCE -DENABLE_ASYNC_PROGRESS
SERVER_TARGET       = ep_server
CLIENT_TARGET       = libep

ifeq ($(CC), icc)
  LDFLAGS += -static-intel
  SERVER_FLAGS += -wd266
  CLIENT_FLAGS += -diag-disable=1478 -wd266
else ifeq ($(CC), gcc)
  CLIENT_FLAGS += -Wno-deprecated-declarations
endif

ifeq ($(MPIRT),$(filter $(MPIRT), intel openmpi))
    LDFLAGS += -L$(MPIRT_DIR)/lib -lmpi -ldl -lrt -lpthread -lm
else ifeq ($(MPIRT), cray)
    LDFLAGS += -L$(MPIRT_DIR)/lib -lmpich -ldl -lrt -lpthread -lm
endif

ifneq ($(PEERCOMM), 0)
  CLIENT_FLAGS += -DENABLE_PEERCOMM
endif

ifeq ($(ATOMIC), 1)
  CLIENT_FLAGS += -DENABLE_CQUEUE_ATOMIC
  SERVER_FLAGS += -DENABLE_CQUEUE_ATOMIC
endif

ifeq ($(FILEIO), 1)
  CLIENT_FLAGS += -DENABLE_FILEIO
  SERVER_FLAGS += -DENABLE_FILEIO
endif

default: $(SERVER_TARGET) $(CLIENT_TARGET)

$(SERVER_TARGET): cqueue.c memory.c server.c env.c sig_handler.c $(QUANT_DIR)/quant.c allreduce_pr.c
	$(CC) $(SERVER_FLAGS) -c cqueue.c
	$(CC) $(SERVER_FLAGS) -c memory.c
	$(CC) $(SERVER_FLAGS) -c env.c
	$(CC) $(SERVER_FLAGS) -c sig_handler.c
	$(CC) $(SERVER_FLAGS) -c $(QUANT_DIR)/quant.c
	$(CC) $(SERVER_FLAGS) -c allreduce_pr.c
	$(CC) $(SERVER_FLAGS) -o $(SERVER_TARGET) server.c cqueue.o memory.o env.o sig_handler.o quant.o allreduce_pr.o $(LDFLAGS)

$(CLIENT_TARGET): handle.c window.c cqueue.c memory.c client.c wrapper.c env.c sig_handler.c uuid.c server.c $(QUANT_DIR)/quant.c allreduce_pr.c
	$(CC) $(CLIENT_FLAGS) -c handle.c
	$(CC) $(CLIENT_FLAGS) -c window.c
	$(CC) $(SERVER_FLAGS) -c allreduce_pr.c
	$(CC) $(CLIENT_FLAGS) -c cqueue.c
	$(CC) $(CLIENT_FLAGS) -c memory.c
	$(CC) $(CLIENT_FLAGS) -c client.c
	$(CC) $(CLIENT_FLAGS) -c wrapper.c
	$(CC) $(CLIENT_FLAGS) -c env.c
	$(CC) $(CLIENT_FLAGS) -c sig_handler.c
	$(CC) $(CLIENT_FLAGS) -c uuid.c
	$(CC) $(CLIENT_FLAGS) -c server.c
	$(CC) $(CLIENT_FLAGS) -c $(QUANT_DIR)/quant.c
	$(CC) $(CLIENT_FLAGS) -c allreduce_pr.c
    # TODO: need to link libmpi.a
	ar -cvq $(CLIENT_TARGET).a window.o handle.o cqueue.o memory.o client.o wrapper.o env.o sig_handler.o uuid.o server.o quant.o allreduce_pr.o
	$(CC) $(CLIENT_FLAGS) -shared -Wl,-soname,$(CLIENT_TARGET).so.1 -o $(CLIENT_TARGET).so.1.0 *.o $(LDFLAGS)

test:
	cd tests; make clean; make

clean:
	rm -f $(SERVER_TARGET) *.a *.so *.so.1.0 *.o
