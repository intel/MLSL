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
#include <dirent.h>
#include <stdlib.h>
#include <string.h>

#include "log.hpp"
#include "mlsl.hpp"
#include "sysinfo.hpp"

#define CPUINFO_PATH     "/proc/cpuinfo"
#define INFINIBAND_PATH  "/sys/class/infiniband/"
#define HFI_FILENAME     "hfi"
#define MLX_FILENAME     "mlx"
#define HFI_FILENAME_LEN strlen(HFI_FILENAME)
#define MLX_FILENAME_LEN strlen(MLX_FILENAME)
#define MAX_STR_LEN      200

namespace MLSL
{
    string ConvertToCpuName(CpuType cpuType)
    {
        switch (cpuType)
        {
            case CPU_NONE: return "CPU_NONE";
            case XEON:     return "Xeon";
            case XEON_PHI: return "Xeon Phi";
            default:       MLSL_ASSERT(0, "Invalid cpuType: %d", cpuType);
        }
    }

    string ConvertToNetName(NetType netType)
    {
        switch (netType)
        {
            case NET_NONE: return "NET_NONE";
            case ETH:      return "Ethernet";
            case MLX:      return "Infiniband";
            case HFI:      return "Omni-Path";
            default:       MLSL_ASSERT(0, "Invalid netType: %d", netType);
        }
    }

    void SysInfo::GetCpuPropetyValue(Property property, char* propertyValue, FILE* file)
    {
        char fullStr[MAX_STR_LEN];
        const char* propertyName = property.name.c_str();
        size_t propertyNameLength = strlen(propertyName);

        while (fgets(fullStr, MAX_STR_LEN, file))
        {
            if (strncmp(fullStr, propertyName, propertyNameLength) == 0)
            {
                char* tmpValue = NULL;
                strtok(fullStr, ":");
                tmpValue = strtok(NULL, "\n");

                MLSL_ASSERT(tmpValue, "Invalid file format");

                size_t countToCopy = strlen(tmpValue);
                size_t offset = 0;
                while (tmpValue[offset] == ' ') offset++;

                strcpy(propertyValue, &tmpValue[offset]);
                propertyValue[countToCopy + 1] = '\0';
                return;
            }
        }
        MLSL_ASSERT(0, "Invalid file format or property");
    }

    CpuType SysInfo::ConvertToCpuType(char* name)
    {
        string cpuName(name);
        if (cpuName.find("Phi") != string::npos)
            return XEON_PHI;
        else if (cpuName.find("Xeon") != string::npos)
            return XEON;
        return CPU_NONE;
    }

    NetType SysInfo::ConvertToNetType(char* name)
    {
        if (name == NULL)
            return ETH;
        else
        {
            if (strncmp(name, MLX_FILENAME, MLX_FILENAME_LEN) == 0)
                return MLX;
            else
            {
                if (strncmp(name, HFI_FILENAME, HFI_FILENAME_LEN) == 0)
                    return HFI;
                else
                    return NET_NONE;
            }
        }
    }

    void SysInfo::SetCpuInfo()
    {
        char propertyValue[MAX_STR_LEN];
        Property propertyNames[3];
        propertyNames[0].name = "model name";
        propertyNames[1].name = "cpu cores";
        propertyNames[2].name = "siblings";

        FILE* cpuInfoFile = fopen(CPUINFO_PATH, "r");

        MLSL_ASSERT(cpuInfoFile, "Can't open file: %s", CPUINFO_PATH);

        GetCpuPropetyValue(propertyNames[0], propertyValue, cpuInfoFile);
        cpuInfo.cpuType = ConvertToCpuType(propertyValue);

        fseek(cpuInfoFile, 0, SEEK_SET);
        GetCpuPropetyValue(propertyNames[1], propertyValue, cpuInfoFile);
        cpuInfo.cpuCoresCount = atoi(propertyValue);

        fseek(cpuInfoFile, 0, SEEK_SET);
        GetCpuPropetyValue(propertyNames[2], propertyValue, cpuInfoFile);
        cpuInfo.cpuThreadsCount = atoi(propertyValue);

        fclose(cpuInfoFile);
    }

    void SysInfo::SetNetInfo()
    {
        netInfo.clear();
        DIR* dir = opendir(INFINIBAND_PATH);
        NetInfo nInfo;
        nInfo.netType = NET_NONE;
        if (dir == NULL)
        {
            nInfo.netType = ConvertToNetType(NULL);
            netInfo.push_back(nInfo);
        }
        else
        {
            struct dirent* object;
            while ((object = readdir(dir)) != NULL)
            {
                MLSL_LOG(DEBUG, "d_name %s, d_type %d", object->d_name, object->d_type);
                nInfo.netType = ConvertToNetType(object->d_name);
                if (nInfo.netType != NET_NONE)
                    netInfo.push_back(nInfo);

            }

            for (size_t i = 0; i < netInfo.size(); i++)
            {
                if (netInfo[i].netType == HFI)
                {
                    swap(netInfo[0], netInfo[i]);
                    break;
                }
            }
            closedir(dir);
        }

        if (netInfo.empty()) netInfo.push_back(nInfo);
    }

    void SysInfo::Initialize()
    {
        SetCpuInfo();
        SetNetInfo();
    }

    SysInfo::SysInfo()
    {
        cpuInfo.cpuType = CPU_NONE;
        cpuInfo.cpuCoresCount = 0;
        cpuInfo.cpuThreadsCount = 0;
        NetInfo defaultNetInfo;
        defaultNetInfo.netType = NET_NONE;
        netInfo.push_back(defaultNetInfo);
    }

    SysInfo::~SysInfo()
    {
        netInfo.clear();
    }

    NetInfo SysInfo::GetNetInfo()
    {
        MLSL_ASSERT(!netInfo.empty(), "no net information");
        return netInfo[0];
    }
}
