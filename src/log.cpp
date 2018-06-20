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
    int Env2Int(const char* envName, int* val)
    {
        const char* valPtr;

        valPtr = getenv(envName);
        if (valPtr)
        {
            const char* p;
            int sign = 1, value = 0;
            p = valPtr;
            while (*p && IS_SPACE(*p))
                p++;
            if (*p == '-')
            {
                p++;
                sign = -1;
            }
            if (*p == '+')
                p++;

            while (*p && isdigit(*p))
                value = 10 * value + (*p++ - '0');

            if (*p)
            {
                MLSL_LOG(ERROR, "invalid character %c in %s", *p, envName);
                return -1;
            }
            *val = sign * value;
            return 1;
        }
        return 0;
    }

    void GetTime(char* buf, size_t bufSize)
    {
        time_t timer;
        struct tm* timeInfo = 0;
        time(&timer);
        timeInfo = localtime(&timer);
        assert(timeInfo);
        strftime(buf, bufSize, "%Y:%m:%d %H:%M:%S", timeInfo);
    }

    void PrintBacktrace(void)
    {
        int j, nptrs;
        void* buffer[100];
        char** strings;

        nptrs = backtrace(buffer, 100);
        printf("backtrace() returned %d addresses\n", nptrs);
        fflush(stdout);

        strings = backtrace_symbols(buffer, nptrs);
        if (strings == NULL)
        {
            perror("backtrace_symbols");
            exit(EXIT_FAILURE);
        }

        for (j = 0; j < nptrs; j++)
        {
            printf("%s\n", strings[j]);
            fflush(stdout);
        }
        free(strings);
    }
}
