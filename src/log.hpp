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
#ifndef LOG_HPP
#define LOG_HPP

#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "env.hpp"

#define GET_TID()    syscall(SYS_gettid)
#define IS_SPACE(c)  ((c==0x20 || c==0x09 || c==0x0a || c==0x0b || c==0x0c || c==0x0d) ? 8 : 0)
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define MLSL_LOG(log_lvl, fmt, ...)                                                    \
  do {                                                                                 \
        if (log_lvl <= envData.logLevel)                                               \
        {                                                                              \
            char time_buf[20]; /*2016:07:21 14:47:39*/                                 \
            GetTime(time_buf, 20);                                                     \
            switch (log_lvl)                                                           \
            {                                                                          \
                case ERROR:                                                            \
                {                                                                      \
                    printf("%s: ERROR: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),   \
                            __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
                    PrintBacktrace();                                                  \
                    break;                                                             \
                }                                                                      \
                case INFO:                                                             \
                {                                                                      \
                    printf("(%ld): %s:%u " fmt "\n", GET_TID(),                        \
                            __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
                    break;                                                             \
                }                                                                      \
                case DEBUG:                                                            \
                case TRACE:                                                            \
                {                                                                      \
                    printf("%s: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),          \
                            __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
                    break;                                                             \
                }                                                                      \
                default:                                                               \
                {                                                                      \
                    assert(0);                                                         \
                }                                                                      \
            }                                                                          \
            fflush(stdout);                                                            \
        }                                                                              \
  } while (0)

#define MLSL_ASSERT(cond, fmt, ...)                                                       \
  do                                                                                      \
  {                                                                                       \
      if (!(cond))                                                                        \
      {                                                                                   \
          fprintf(stderr, "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n",               \
                  GET_TID(), __FILENAME__, __FUNCTION__, __LINE__, #cond, ##__VA_ARGS__); \
          fflush(stderr);                                                                 \
          MLSL::Environment::GetEnv().Finalize();                                         \
          _exit(1);                                                                       \
      }                                                                                   \
  } while(0)

namespace MLSL
{
    enum LogLevel
    {
        ERROR = 0,
        INFO,
        DEBUG,
        TRACE
    };

    int Env2Int(const char* env_name, int* val);
    void GetTime(char* buf, size_t buf_size);
    void PrintBacktrace(void);
}

#endif /* LOG_HPP */
