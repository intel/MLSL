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
MLSL_YEAR                   = 2018
MLSL_MAJOR_VERSION          = 2018
MLSL_MINOR_VERSION          = 2
MLSL_PACKAGE_ID             = 006
MLSL_FULL_VERSION           = $(MLSL_MAJOR_VERSION).$(MLSL_MINOR_VERSION).$(MLSL_PACKAGE_ID)
# Examples: "2017 Beta", "2017", "2017 Update 1"
ifeq (true, $(shell git rev-parse --is-inside-work-tree 2>/dev/null))
    MLSL_GIT_VERSION        = `git log --pretty=format:"%h" -1`
else
    MLSL_GIT_VERSION        = UNKNOWN
endif
MLSL_OFFICIAL_VERSION       = $(MLSL_MAJOR_VERSION) Update 2 Preview
MLSL_COPYRIGHT_YEAR         = `date +%Y`
MLSL_ARCHIVE_PREFIX         = l_mlsl_$(MLSL_FULL_VERSION)
MLSL_ARCHIVE_SUFFIX         ?=
MLSL_ARCHIVE_NAME           = $(MLSL_ARCHIVE_PREFIX)$(MLSL_ARCHIVE_SUFFIX).tgz
LIBMLSL_SONAME              = libmlsl.so.1
LIBMLSL_SO_FILENAME         = libmlsl.so.1.0
SHELL=bash
SHELL = bash
COMPILER ?= gnu
MLSL_MODE ?= process
#COMPILER ?= intel
# possible values: intel, openmpi, cray
MPIRT     ?= intel
MPIRT_DIR_2018 ?= $(BASE_DIR)/mpirt
MPIRT_DIR_2019 ?= $(BASE_DIR)/mpirt_2019
ifeq ($(MLSL_MODE),thread)
	MPIRT_DIR = $(BASE_DIR)/mpirt_2019
	COMM_EP            = 0
	COMM_HANDOFF       = 1
else
	MPIRT_DIR = $(BASE_DIR)/mpirt
	COMM_EP            = 1
	COMM_HANDOFF       = 0
endif

USE_SECURITY_FLAGS ?= 1
ENABLE_DEBUG       ?= 0
ENABLE_CHKP        ?= 0
ENABLE_CHKP_INT    ?= 0
ENABLE_MT_MEMCPY   ?= 0
ENABLE_INTERNAL_ENV_UPDATE ?= 0

ifeq ($(COMM_EP),1)
    ifeq ($(COMM_HANDOFF),1)
        $(error Specify only one knob: COMM_EP or COMM_HANDOFF)
    endif
endif

# will be propagated to other modules
EXTRA_CFLAGS  = 
EXTRA_LDFLAGS = 

ifeq ($(ENABLE_CHKP), 1)
    CHKP_FLAGS = -check-pointers=rw -check-pointers-dangling=all -check-pointers-undimensioned -check-pointers-narrowing -rdynamic
    EXTRA_CFLAGS  += $(CHKP_FLAGS) -DENABLE_CHKP=1
    EXTRA_LDFLAGS += $(CHKP_FLAGS)
endif

EXTRA_CFLAGS += -Wall -Werror

CXXFLAGS += $(EXTRA_CFLAGS)
LDFLAGS  += $(EXTRA_LDFLAGS)

ARX86       = ar
CXXFLAGS    += -fPIC
INCS        += -I$(INCLUDE_DIR) -I$(SRC_DIR)

ifeq ($(COMPILER), intel)
    CC        = icc
    CXX       = icpc
    CXXFLAGS += -std=c++11
    LDFLAGS  += -static-intel
else ifeq ($(COMPILER), gnu)
    CC        = gcc
    CXX       = g++
    CXXFLAGS += -std=c++0x
else
    $(error Unsupported compiler $(COMPILER))
endif

ifeq ($(ENABLE_DEBUG), 1)
    USE_SECURITY_FLAGS = 0
    CXXFLAGS += -O0 -g
else
    CXXFLAGS += -O2
endif

ifeq ($(CODECOV),1)
    CXXFLAGS += -prof-gen=srcpos -prof-src-root-cwd
endif

EPLIB_TARGET        = libep
TARGET              = libmlsl
TESTS_TARGET        = tests
SRC_DIR             = $(BASE_DIR)/src
INCLUDE_DIR         = $(BASE_DIR)/include
SCRIPTS_DIR         = $(BASE_DIR)/scripts
DOC_DIR             = $(BASE_DIR)/doc
TESTS_DIR           = $(BASE_DIR)/tests
EXAMPLES_DIR        = $(TESTS_DIR)/examples
EPLIB_DIR           = $(BASE_DIR)/eplib
QUANT_DIR           = $(BASE_DIR)/quant
DOXYGEN_DIR         = $(BASE_DIR)/doc/doxygen
PREFIX              ?= $(BASE_DIR)/_install
TMP_COVERAGE_DIR    ?= $(BASE_DIR)
CODECOV_SRCROOT     ?= $(BASE_DIR)
INTEL64_PREFIX      = $(PREFIX)/intel64
STAGING             = $(BASE_DIR)/_staging
INTEL64_STAGING     = $(STAGING)/intel64
TMP_DIR             = $(BASE_DIR)/_tmp
TMP_ARCHIVE_DIR     = $(TMP_DIR)/$(MLSL_ARCHIVE_PREFIX)$(MLSL_ARCHIVE_SUFFIX)
CXXFLAGS            += -I$(MPIRT_DIR)/include

ifeq ($(MPIRT),$(filter $(MPIRT), intel openmpi))
	LDFLAGS             += -L$(MPIRT_DIR)/lib -lmpi -ldl -lrt -lpthread
else ifeq ($(MPIRT), cray)
    LDFLAGS             += -L$(MPIRT_DIR)/lib -lmpich -ldl -lrt -lpthread
endif

ifeq ($(USE_SECURITY_FLAGS),1)
    SECURITY_CXXFLAGS   = -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector
    SECURITY_LDFLAGS    = -z noexecstack -z relro -z now
    CXXFLAGS            += $(SECURITY_CXXFLAGS)
    LDFLAGS             += $(SECURITY_LDFLAGS)
endif

CXXFLAGS += -DMLSL_YEAR=$(MLSL_YEAR) -DMLSL_FULL_VERSION=$(MLSL_FULL_VERSION) -DMLSL_OFFICIAL_VERSION="$(MLSL_OFFICIAL_VERSION)" -DMLSL_GIT_VERSION="STR($(MLSL_GIT_VERSION))"

SRCS += src/mlsl.cpp
SRCS += src/mlsl_impl.cpp
SRCS += src/mlsl_impl_stats.cpp
SRCS += src/log.cpp
SRCS += src/env.cpp
SRCS += src/c_bind.cpp
SRCS += src/sysinfo.cpp

ifeq ($(ENABLE_CHKP_INT), 1)
    SRCS += src/pointer_checker.cpp
endif

ifeq ($(COMM_EP),1)
    INCS        += -I$(EPLIB_DIR) -I$(QUANT_DIR)
    EPLIB_OBJS  := $(EPLIB_DIR)/*.o
    LIBS        += $(EPLIB_OBJS)
    SRCS        += src/comm_ep.cpp
    USE_EPLIB    = 1
    ifeq ($(BASEFAST),1)
        ENABLE_BASEMPI_COMM_FAST=1
    endif
else ifeq ($(COMM_HANDOFF),1)
    SRCS        += src/comm_handoff.cpp
    ifeq ($(ENABLE_INTERNAL_ENV_UPDATE),1)
        CXXFLAGS += -DINTERNAL_ENV_UPDATE
    endif
endif

OBJS := $(SRCS:.cpp=.o)

yesnolist += ENABLE_DEBUG
yesnolist += ENABLE_CHKP_INT
yesnolist += USE_EPLIB

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

ifneq ($(strip $(DEFS)),)
    $(info Using Options:$(DEFS))
endif

all:
	$(MAKE) clean
	$(MAKE) $(TARGET) MLSL_MODE="thread"
	$(MAKE) clean
	$(MAKE) $(EPLIB_TARGET) MLSL_MODE="process"
	$(MAKE) $(TARGET) MLSL_MODE="process"

dev:
	$(MAKE) clean && $(MAKE) all && $(MAKE) install

testing:
	$(MAKE) -s -C $(EXAMPLES_DIR) MLSL_MODE="$(MLSL_MODE)" CC="$(CC)" CXX="$(CXX)" EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" ENABLE_DEBUG=$(ENABLE_DEBUG) && $(MAKE) -s -C $(EXAMPLES_DIR) run

examples:
	$(MAKE) -C $(EXAMPLES_DIR) MLSL_MODE="$(MLSL_MODE)" CC="$(CC)" CXX="$(CXX)" MLSL_ROOT=$(PREFIX) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" ENABLE_DEBUG=$(ENABLE_DEBUG)

$(CUSTOM_MAKE_FILE): checkoptions

.lastoptions: checkoptions

checkoptions: ;
	@echo $(DEFS) > .curoptions
	@if ! test -f .lastoptions || ! diff .curoptions .lastoptions > /dev/null ; then mv -f .curoptions .lastoptions ; fi
	@rm -f .curoptions

$(TARGET): src/libmlsl.a src/$(LIBMLSL_SO_FILENAME)
	mkdir -p src/$(MLSL_MODE)
	cp src/libmlsl.a  src/$(MLSL_MODE)
	cp src/$(LIBMLSL_SO_FILENAME) src/$(MLSL_MODE)

src/libmlsl.a: $(OBJS)
	$(ARX86) rcs $(SRC_DIR)/$(TARGET).a $(OBJS) $(EPLIB_OBJS)

src/$(LIBMLSL_SO_FILENAME): $(OBJS)
		LD_LIBRARY_PATH=$(MPIRT_DIR)/lib $(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -shared -Wl,-soname,$(LIBMLSL_SONAME) -o $(SRC_DIR)/$(LIBMLSL_SO_FILENAME) $(OBJS) $(LDFLAGS)

src/%.o: $(SRC_DIR)/%.cpp .lastoptions
	$(CXX) -c $(CXXFLAGS) $(DEFS) $(INCS) $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.d $(SRC_DIR)/$(LIBMLSL_SO_FILENAME) $(SRC_DIR)/*.a .lastoptions *.tgz *.log
	rm -f CODE_COVERAGE.HTML codecov.xml coverage.xml *.dpi *.dpi.lock *.dyn *.spi *.spl
	rm -rf $(STAGING) $(TMP_DIR) CodeCoverage coverage
	cd $(EPLIB_DIR) && $(MAKE) clean
	cd $(EXAMPLES_DIR) && $(MAKE) clean

cleanall: clean

-include $(SRCS:.cpp=.d)

$(EPLIB_TARGET): eplib/ep_server eplib/libep.a

eplib/ep_server:
	cd $(EPLIB_DIR) && LD_LIBRARY_PATH=$(MPIRT_DIR)/lib  $(MAKE) MLSL_MODE="$(MLSL_MODE)" CC="$(CC)" EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" \
	                           ENABLE_DEBUG=$(ENABLE_DEBUG) USE_SECURITY_FLAGS=$(USE_SECURITY_FLAGS) \
	                           ENABLE_CHKP=$(ENABLE_CHKP) MPIRT="$(MPIRT)" MPIRT_DIR="$(MPIRT_DIR)" \
	                           ep_server

eplib/libep.a:
	cd $(EPLIB_DIR) && $(MAKE) MLSL_MODE="$(MLSL_MODE)" CC="$(CC)" EXTRA_CFLAGS="$(EXTRA_CFLAGS)" EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" \
	                           ENABLE_DEBUG=$(ENABLE_DEBUG) USE_SECURITY_FLAGS=$(USE_SECURITY_FLAGS) \
	                           ENABLE_CHKP=$(ENABLE_CHKP) MPIRT="$(MPIRT)" MPIRT_DIR="$(MPIRT_DIR)" \
	                           libep

archive: _staging
	rm -f $(BASE_DIR)/$(MLSL_ARCHIVE_NAME)
	rm -rf $(TMP_DIR)
	mkdir -p $(TMP_DIR)
	mkdir -p $(TMP_ARCHIVE_DIR)
	cd $(STAGING) && tar cfz $(TMP_ARCHIVE_DIR)/files.tar.gz ./*
	cp $(SCRIPTS_DIR)/install.sh $(TMP_ARCHIVE_DIR)/
	sed -i -e "s|MLSL_SUBSTITUTE_FULL_VERSION|$(MLSL_FULL_VERSION)|g" $(TMP_ARCHIVE_DIR)/install.sh
	sed -i -e "s|MLSL_SUBSTITUTE_OFFICIAL_VERSION|$(MLSL_OFFICIAL_VERSION)|g" $(TMP_ARCHIVE_DIR)/install.sh
	cd $(TMP_DIR) && tar cfz $(BASE_DIR)/$(MLSL_ARCHIVE_NAME) $(MLSL_ARCHIVE_PREFIX)$(MLSL_ARCHIVE_SUFFIX)
	rm -rf $(TMP_DIR)
	rm -rf $(STAGING)

install: _staging
	rm -rf $(PREFIX)
	cp -pr $(STAGING) $(PREFIX)
	sed -i -e "s|MLSL_SUBSTITUTE_INSTALLDIR|$(PREFIX)|g" $(INTEL64_PREFIX)/bin/mlslvars.sh
	if [ "$(MPIRT)" == "intel" ]; then \
		sed -i -e "s|I_MPI_SUBSTITUTE_INSTALLDIR|$(PREFIX)|g" $(INTEL64_PREFIX)/etc/mpiexec.conf; \
	fi
	rm -rf $(STAGING)			  

doxygen:
	cd $(DOXYGEN_DIR) && doxygen $(DOXYGEN_DIR)/doxygen.config

_staging:
	rm -rf $(STAGING)
	for mode in process thread ; do \
		mkdir -p $(INTEL64_STAGING)/bin/$$mode ; \
		mkdir -p $(INTEL64_STAGING)/lib/$$mode ; \
		mkdir -p $(INTEL64_STAGING)/etc ; \
		mkdir -p $(STAGING)/licensing/mpi/$$mode ; \
	done ;
	mkdir -p $(INTEL64_STAGING)/include
	mkdir -p $(INTEL64_STAGING)/include/mlsl
	mkdir -p $(STAGING)/test
	mkdir -p $(STAGING)/example
	mkdir -p $(STAGING)/licensing/mlsl
	mkdir -p $(STAGING)/doc
	mkdir -p $(STAGING)/doc/api
	cp $(EPLIB_DIR)/ep_server $(INTEL64_STAGING)/bin
	for mode in process thread ; do \
		cp $(SRC_DIR)/$$mode/$(TARGET).so.1.0 $(INTEL64_STAGING)/lib/$$mode ; \
		cd $(INTEL64_STAGING)/lib/$$mode && ln -s $(TARGET).so.1.0 $(TARGET).so.1 ; \
		cd $(INTEL64_STAGING)/lib/$$mode && ln -s $(TARGET).so.1 $(TARGET).so ; \
	done ;
	cp $(INCLUDE_DIR)/mlsl.hpp $(INTEL64_STAGING)/include
	cp $(INCLUDE_DIR)/mlsl.h $(INTEL64_STAGING)/include
	cp $(INCLUDE_DIR)/mlsl/mlsl.py $(INTEL64_STAGING)/include/mlsl
	cp $(INCLUDE_DIR)/mlsl/__init__.py $(INTEL64_STAGING)/include/mlsl
	cp $(SCRIPTS_DIR)/mlslvars.sh $(INTEL64_STAGING)/bin
	if [ "$(MPIRT)" == "intel" ]; then \
		cp $(MPIRT_DIR_2018)/bin/mpiexec.hydra $(INTEL64_STAGING)/bin/process; \
		cp $(MPIRT_DIR_2018)/bin/pmi_proxy $(INTEL64_STAGING)/bin/process; \
		cp $(MPIRT_DIR_2018)/bin/hydra_persist $(INTEL64_STAGING)/bin/process; \
		cd $(INTEL64_STAGING)/bin/process && ln -s mpiexec.hydra mpiexec; \
		cp $(MPIRT_DIR_2018)/lib/libmpi.so.12.0 $(INTEL64_STAGING)/lib/process; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libmpi.so.12.0 libmpi.so.12; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libmpi.so.12.0 libmpi.so; \
		cp $(MPIRT_DIR_2018)/lib/libtmi.so.1.2 $(INTEL64_STAGING)/lib/process; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmi.so.1.2 libtmi.so; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmi.so.1.2 libtmi.so.1.0; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmi.so.1.2 libtmi.so.1.1; \
		cp $(MPIRT_DIR_2018)/lib/libtmip_psm.so.1.2 $(INTEL64_STAGING)/lib/process; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmip_psm.so.1.2 libtmip_psm.so; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmip_psm.so.1.2 libtmip_psm.so.1.0; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmip_psm.so.1.2 libtmip_psm.so.1.1; \
		cp $(MPIRT_DIR_2018)/lib/libtmip_psm2.so.1.0 $(INTEL64_STAGING)/lib/process; \
		cd $(INTEL64_STAGING)/lib/process && ln -s libtmip_psm2.so.1.0 libtmip_psm2.so; \
		cp $(MPIRT_DIR_2018)/etc/mpiexec.conf $(INTEL64_STAGING)/etc; \
		cp $(MPIRT_DIR_2018)/etc/tmi.conf $(INTEL64_STAGING)/etc; \
		cp $(MPIRT_DIR_2018)/licensing/license.txt $(STAGING)/licensing/mpi/process/; \
		cp $(MPIRT_DIR_2018)/licensing/third-party-programs.txt $(STAGING)/licensing/mpi/process/; \
		cp $(MPIRT_DIR_2018)/bin/mpirun $(INTEL64_STAGING)/bin/process; \
		cd $(INTEL64_STAGING)/lib/ && ln -s process/libmlsl.so.1.0 libmlsl.so; \
		cd $(INTEL64_STAGING)/lib/ && ln -s process/libmlsl.so.1.0 libmlsl.so.1; \
		cp $(MPIRT_DIR_2019)/bin/fi_info $(INTEL64_STAGING)/bin/thread; \
		cp $(MPIRT_DIR_2019)/bin/mpiexec.hydra $(INTEL64_STAGING)/bin/thread; \
		cp $(MPIRT_DIR_2019)/bin/hydra_pmi_proxy $(INTEL64_STAGING)/bin/thread; \
		cp $(MPIRT_DIR_2019)/bin/hydra_nameserver $(INTEL64_STAGING)/bin/thread; \
		cp $(MPIRT_DIR_2019)/bin/hydra_bstrap_proxy $(INTEL64_STAGING)/bin/thread; \
		cd $(INTEL64_STAGING)/bin/thread && ln -s mpiexec.hydra mpiexec; \
		cp $(MPIRT_DIR_2019)/lib/libmpi.so.12.0.0 $(INTEL64_STAGING)/lib/thread; \
		cp $(MPIRT_DIR_2019)/lib/libfabric.so.1 $(INTEL64_STAGING)/lib/thread; \
		cp -R $(MPIRT_DIR_2019)/lib/prov/ $(INTEL64_STAGING)/lib/thread; \
		cp $(MPIRT_DIR_2019)/etc/* $(INTEL64_STAGING)/etc; \
		cd $(INTEL64_STAGING)/lib/thread && ln -s libfabric.so.1 libfabric.so; \
		cd $(INTEL64_STAGING)/lib/thread && ln -s libmpi.so.12.0.0 libmpi.so.12.0; \
		cd $(INTEL64_STAGING)/lib/thread && ln -s libmpi.so.12.0.0 libmpi.so.12; \
		cd $(INTEL64_STAGING)/lib/thread && ln -s libmpi.so.12.0.0 libmpi.so; \
		cp $(MPIRT_DIR_2019)/bin/mpirun $(INTEL64_STAGING)/bin/thread; \
		cp $(MPIRT_DIR_2019)/licensing/license.txt $(STAGING)/licensing/mpi/thread/; \
		cp $(MPIRT_DIR_2019)/licensing/third-party-programs.txt $(STAGING)/licensing/mpi/thread/; \
	fi
	cp $(EXAMPLES_DIR)/mlsl_test/mlsl_test.cpp $(STAGING)/test/; \
	cp $(EXAMPLES_DIR)/mlsl_test/cmlsl_test.c $(STAGING)/test/; \
	cp $(EXAMPLES_DIR)/mlsl_test/mlsl_test.py $(STAGING)/test/; \
	cp $(EXAMPLES_DIR)/mlsl_test/Makefile $(STAGING)/test/; \
	cp $(EXAMPLES_DIR)/mlsl_example/mlsl_example.cpp $(STAGING)/example/; \
	cp $(EXAMPLES_DIR)/mlsl_example/Makefile $(STAGING)/example/; \
	cp $(DOC_DIR)/README.txt $(STAGING)/doc/
	cp $(DOC_DIR)/API_Reference.htm $(STAGING)/doc/
	cp -r $(DOC_DIR)/doxygen/html/* $(STAGING)/doc/api
	cp $(DOC_DIR)/Developer_Guide.pdf $(STAGING)/doc/
	cp $(DOC_DIR)/Release_Notes.txt $(STAGING)/doc/
	sed -i -e "s|MLSL_SUBSTITUTE_OFFICIAL_VERSION|$(MLSL_OFFICIAL_VERSION)|g" $(STAGING)/doc/Release_Notes.txt
	chmod 755 $(STAGING)/doc
	chmod 755 $(STAGING)/doc/api
	chmod 644 $(STAGING)/doc/API_Reference.htm
	find $(STAGING)/doc/api -type d -exec chmod 755 {} \;
	find $(STAGING)/doc/api -type f -exec chmod 644 {} \;
	chmod 644 $(STAGING)/doc/Developer_Guide.pdf
	chmod 644 $(STAGING)/doc/README.txt
	chmod 644 $(STAGING)/doc/Release_Notes.txt
	chmod 755 $(INTEL64_STAGING)
	chmod 755 $(INTEL64_STAGING)/bin
	chmod 755 $(INTEL64_STAGING)/bin/ep_server
	chmod 755 $(INTEL64_STAGING)/bin/mlslvars.sh
	chmod 755 $(INTEL64_STAGING)/include
	chmod 644 $(INTEL64_STAGING)/include/mlsl.hpp
	chmod 644 $(INTEL64_STAGING)/include/mlsl.h
	chmod 755 $(INTEL64_STAGING)/include/mlsl
	chmod 644 $(INTEL64_STAGING)/include/mlsl/mlsl.py
	chmod 644 $(INTEL64_STAGING)/include/mlsl/__init__.py
	chmod 755 $(INTEL64_STAGING)/lib
	chmod 755 $(INTEL64_STAGING)/lib/process/libmlsl.so
	chmod 755 $(INTEL64_STAGING)/lib/process/libmlsl.so.1
	chmod 755 $(INTEL64_STAGING)/lib/libmlsl.so
	chmod 755 $(INTEL64_STAGING)/lib/libmlsl.so.1
	chmod 755 $(INTEL64_STAGING)/lib/process/libmlsl.so.1.0
	chmod 755 $(INTEL64_STAGING)/lib/thread/libmlsl.so
	chmod 755 $(INTEL64_STAGING)/lib/thread/libmlsl.so.1
	chmod 755 $(INTEL64_STAGING)/lib/thread/libmlsl.so.1.0
	chmod 755 $(INTEL64_STAGING)/bin/thread/fi_info
	chmod 755 $(STAGING)/test
	chmod 644 $(STAGING)/test/cmlsl_test.c
	chmod 644 $(STAGING)/test/mlsl_test.cpp
	chmod 755 $(STAGING)/test/mlsl_test.py
	chmod 644 $(STAGING)/test/Makefile
	chmod 755 $(STAGING)/licensing
	if [ "$(MPIRT)" == "intel" ]; then \
		chmod 755 $(INTEL64_STAGING)/bin/process/mpiexec; \
		chmod 755 $(INTEL64_STAGING)/bin/process/mpiexec.hydra; \
		chmod 755 $(INTEL64_STAGING)/bin/process/mpirun; \
		chmod 755 $(INTEL64_STAGING)/bin/process/pmi_proxy; \
		chmod 644 $(INTEL64_STAGING)/etc/mpiexec.conf; \
		chmod 644 $(INTEL64_STAGING)/etc/tmi.conf; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libmpi.so; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libmpi.so.12; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libmpi.so.12.0; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm2.so; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm2.so.1.0; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm.so; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm.so.1.0; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm.so.1.1; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmip_psm.so.1.2; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmi.so; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmi.so.1.0; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmi.so.1.1; \
		chmod 755 $(INTEL64_STAGING)/lib/process/libtmi.so.1.2; \
		chmod 755 $(STAGING)/licensing/mpi; \
		chmod 644 $(STAGING)/licensing/mpi/process/license.txt; \
		chmod 644 $(STAGING)/licensing/mpi/process/third-party-programs.txt; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/mpiexec; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/mpiexec.hydra; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/mpirun; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/hydra_pmi_proxy; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/hydra_bstrap_proxy; \
		chmod 755 $(INTEL64_STAGING)/bin/thread/hydra_nameserver; \
		chmod 755 $(INTEL64_STAGING)/lib/thread/libmpi.so; \
		chmod 755 $(INTEL64_STAGING)/lib/thread/libfabric.so; \
		chmod 755 $(INTEL64_STAGING)/lib/thread/libfabric.so.1; \
		chmod 755 $(INTEL64_STAGING)/lib/thread/libmpi.so.12; \
		chmod 755 $(INTEL64_STAGING)/lib/thread/libmpi.so.12.0; \
		chmod 755 $(INTEL64_STAGING)/etc/; \
		chmod 755 $(INTEL64_STAGING)/etc/tuning_skx_shm-ofi.dat; \
		chmod 755 $(INTEL64_STAGING)/etc/tuning_skx_ofi.dat; \
		chmod 755 $(INTEL64_STAGING)/etc/tuning_knl_shm-ofi.dat; \
		chmod 755 $(INTEL64_STAGING)/etc/tuning_knl_ofi.dat; \
		chmod 644 $(STAGING)/licensing/mpi/thread/license.txt; \
		chmod 644 $(STAGING)/licensing/mpi/thread/third-party-programs.txt; \
	fi
	chmod 755 $(STAGING)/licensing/mlsl
	cp $(BASE_DIR)/LICENSE $(STAGING)/licensing/mlsl/
	cp $(BASE_DIR)/third-party-programs.txt $(STAGING)/licensing/mlsl/
	chmod 644 $(STAGING)/licensing/mlsl/LICENSE
	chmod 644 $(STAGING)/licensing/mlsl/third-party-programs.txt

.PHONY: checkoptions examples doxygen dev
