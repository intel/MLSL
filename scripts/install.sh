#!/bin/bash
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

SCRIPT_DIR=`cd $(dirname "$BASH_SOURCE") && pwd -P`
BASENAME=`basename $0`
USERNAME=`whoami`

SILENT_MODE=0

if [ "$USERNAME" = "root" ]
then
    DEFAULT_INSTALL_PATH="/opt/intel/mlsl_MLSL_SUBSTITUTE_FULL_VERSION"
else
    DEFAULT_INSTALL_PATH="${HOME}/intel/mlsl_MLSL_SUBSTITUTE_FULL_VERSION"
fi

print_help()
{
    echo "Usage: $BASENAME [-d <install_directory>][-h][-s]"
    echo " -d <install_directory> : install into the specified directory"
    echo " -s                     : install silently"
    echo " -h                     : print help"
}

while [ $# -ne 0 ]
do
    case $1 in
        '-d')
            INSTALL_PATH="$2"
            shift
            ;;
        '-s')
            SILENT_MODE=1
            ;;
        '-h')
            print_help
            exit 0
            ;;
        *)
            echo "ERROR: unknown option ($1)"
            print_help
            exit 1
            ;;
    esac
    
    shift
done

if [ ${SILENT_MODE} -eq 0 ]
then
    echo "Intel(R) Machine Learning Scaling Library MLSL_SUBSTITUTE_OFFICIAL_VERSION for Linux* OS will be installed"
    echo "Type 'y' to continue or 'q' to exit and then press Enter"
    CONFIRMED=0
    while [ $CONFIRMED = 0 ]
    do
        read -e ANSWER
        case $ANSWER in
            'q')
                exit 0
                ;;
            'y')
                CONFIRMED=1
                ;;
            *)
                ;;
        esac
    done
fi

if [ -z "${INSTALL_PATH}" ]
then
    echo "Please specify the installation directory or press Enter to install into the default path [${DEFAULT_INSTALL_PATH}]:"
    read -e INSTALL_PATH

    if [ -z "${INSTALL_PATH}" ]
    then
        INSTALL_PATH=${DEFAULT_INSTALL_PATH}
    fi
fi

if [ ${SILENT_MODE} -eq 0 ]
then
    echo "Intel(R) Machine Learning Scaling Library MLSL_SUBSTITUTE_OFFICIAL_VERSION for Linux* OS will be installed into ${INSTALL_PATH}"
fi

if [ -d ${INSTALL_PATH} ]
then
    if [ ${SILENT_MODE} -eq 0 ]
    then
        echo "WARNING: ${INSTALL_PATH} exists and will be removed"
    fi
    
    rm -rf ${INSTALL_PATH}
    if [ $? -ne 0 ]
    then
        echo "ERROR: unable to clean ${INSTALL_PATH}"
        echo "Please verify the folder permissions and check it isn't in use"
        exit 1
    fi
fi

mkdir -p ${INSTALL_PATH}
if [ $? -ne 0 ]
then
    echo "ERROR: unable to create ${INSTALL_PATH}"
    echo "Please verify the folder permissions"
    exit 1
fi

cd ${INSTALL_PATH} && tar xfzm ${SCRIPT_DIR}/files.tar.gz
if [ $? -ne 0 ]
then
    echo "ERROR: unable to unpack ${SCRIPT_DIR}/files.tar.gz"
    exit 1
fi

sed -i -e "s|MLSL_SUBSTITUTE_INSTALLDIR|${INSTALL_PATH}|g" ${INSTALL_PATH}/intel64/bin/mlslvars.sh
sed -i -e "s|I_MPI_SUBSTITUTE_INSTALLDIR|${INSTALL_PATH}|g" ${INSTALL_PATH}/intel64/etc/mpiexec.conf  
