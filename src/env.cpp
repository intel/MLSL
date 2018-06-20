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
#include "env.hpp"
#include "log.hpp"

namespace MLSL
{
    EnvData envData = { ERROR /* log_level */,
                        1     /* dup_group */,
                        0     /* enable_stats */,
                        1     /* autoConfigType */};

    void ParseEnvVars()
    {
        Env2Int("MLSL_LOG_LEVEL", (int*)&envData.logLevel);
        Env2Int("MLSL_DUP_GROUP", &envData.dupGroup);
        Env2Int("MLSL_STATS", &envData.enableStats);
        Env2Int("MLSL_AUTO_CONFIG_TYPE", &envData.autoConfigType);
    }

    void PrintEnvVars()
    {
        MLSL_LOG(INFO, "MLSL_LOG_LEVEL: %d", envData.logLevel);
        MLSL_LOG(INFO, "MLSL_DUP_GROUP (duplicate process group per distribution): %d", envData.dupGroup);
        MLSL_LOG(INFO, "MLSL_STATS: %d", envData.enableStats);
        MLSL_LOG(INFO, "MLSL_AUTO_CONFIG_TYPE: %d", envData.autoConfigType);
    }
}
