/*
 Copyright 2016-2018 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#ifndef MLSL_SYSINFO_HPP
#define MLSL_SYSINFO_HPP

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

namespace MLSL
{
    enum CpuType
    {
        CPU_NONE = 0,
        XEON     = 1,
        XEON_PHI = 2
    };

    enum NetType
    {
        NET_NONE = 0,
        ETH      = 1,
        MLX      = 2,
        HFI      = 3
    };

    enum AutoConfigType
    {
        NO_AUTO_CONFIG          = 0,
        NET_AUTO_CONFIG         = 1,
        CPU_AUTO_CONFIG         = 2,
        NET_AND_CPU_AUTO_CONFIG = 3
    };

    struct CpuInfo
    {
        enum CpuType cpuType;
        size_t cpuCoresCount;
        size_t cpuThreadsCount;
    };

    struct NetInfo
    {
        enum NetType netType;
    };

    string ConvertToCpuName(CpuType cpuType);
    string ConvertToNetName(NetType netType);

    class SysInfo
    {
    private:
        vector<NetInfo> netInfo;
        CpuInfo cpuInfo;
        void SetCpuInfo();
        void SetNetInfo();
        NetType ConvertToNetType(char* netName);
        CpuType ConvertToCpuType(char* cpuName);
        struct Property { string name; };
        void GetCpuPropetyValue(Property property, char* propertyValue, FILE* file);

    public:
        SysInfo();
        ~SysInfo();
        void Initialize();
        NetInfo GetNetInfo();
        CpuInfo GetCpuInfo() { return cpuInfo;};
    };
}

#endif /*MLSL_SYSINFO_HPP*/
