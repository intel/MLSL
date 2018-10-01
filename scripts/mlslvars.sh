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

print_help()
{
    echo ""
    echo "Usage: mlslvars.sh [mode]"
    echo ""
    echo "mode can be one of the following:"
    echo "      process (default) "
    echo "      thread            "
    echo ""

}

export I_MPI_ROOT="${MLSL_ROOT}"

if [ -z "${PATH}" ]
then
    PATH="${MLSL_ROOT}/intel64/bin"; export PATH
else
    PATH="${MLSL_ROOT}/intel64/bin:${PATH}"; export PATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]
then
    LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib"; export LD_LIBRARY_PATH
else
    LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
fi

if [ -z "${PYTHONPATH}" ]
then
    export PYTHONPATH="${MLSL_ROOT}/intel64/include"
else
    export PYTHONPATH="${MLSL_ROOT}/intel64/include:${PYTHONPATH}"
fi

mode="process"
if [ $# -ne 0 ]
then
    if [ "$1" == "thread" ]; then
        mode="thread"
    fi
fi

case "$mode" in
    "process"|"thread")
        PATH="${MLSL_ROOT}/intel64/bin/process:${PATH}"; export PATH
        LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib/${mode}:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
        if [ "$mode" == "thread" ]; then
            FI_PROVIDER_PATH="${MLSL_ROOT}/intel64/lib/thread/prov"; export FI_PROVIDER_PATH
        fi
        ;;
    -h|--help)
        print_help
        ;;
esac
