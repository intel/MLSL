#!/bin/sh
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

export MLSL_ROOT=MLSL_SUBSTITUTE_INSTALLDIR

if [ -z "${I_MPI_ROOT}" ]
then
    export I_MPI_ROOT="${MLSL_ROOT}"
fi

if [ -z "${PATH}" ]
then
    export PATH="${MLSL_ROOT}/intel64/bin"
else
    export PATH="${MLSL_ROOT}/intel64/bin:${PATH}"
fi

if [ -z "${LD_LIBRARY_PATH}" ]
then
    export LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib"
else
    export LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib:${LD_LIBRARY_PATH}"
fi

if [ -z "${PYTHONPATH}" ]
then
    export PYTHONPATH="${MLSL_ROOT}/intel64/include"
else
    export PYTHONPATH="${MLSL_ROOT}/intel64/include:${PYTHONPATH}"
fi
