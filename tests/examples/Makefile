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

BASE_DIR = $(shell pwd)
ENABLE_DEBUG ?= 1
SHELL = bash
DIRS = $(shell ls -d $(BASE_DIR)/*/)

all:
	for DIR in $(DIRS); \
	do \
		$(MAKE) -C $$DIR CC="$(CC)" CXX="$(CXX)" EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" ENABLE_DEBUG=$(ENABLE_DEBUG) all; \
		if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
			echo "Make FAILED." ; \
		fi ; \
	done;

run:
	for DIR in $(DIRS); \
	do \
		$(MAKE) -C $$DIR run; \
		if [ "$${PIPESTATUS[0]}" -ne 0 ]; then \
			echo "Run FAILED." ; \
		fi ; \
	done;

clean:
	for DIR in $(DIRS); \
	do \
		$(MAKE) -C $$DIR clean; \
	done;

dev:
	make clean && make all && make run
