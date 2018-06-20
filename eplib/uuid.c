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
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "uuid.h"

char* get_uuid()
{
    char uuid[UUID_STR_LEN] = {0};

    int pid;
    long hostid;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long ticks = (long long)tv.tv_sec * 1000000 + (long long)tv.tv_usec;
    hostid = gethostid();
    pid = getpid();

    snprintf(uuid, UUID_STR_LEN,
             "%02hhx%02hhx%02hhx%02hhx-"
             "%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-"
             "%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx",
             (short int)(pid), (short int)(pid >> 8), (short int)(pid >> 16), (short int)(pid >> 24),
             (short int)(ticks),        (short int)(ticks >> 8),
             (short int)(ticks >> 16),  (short int)(ticks >> 24),
             (short int)(ticks >> 32),  (short int)(ticks >> 40),
             (short int)(ticks >> 48),  (short int)(ticks >> 56),
             (short int)(hostid),       (short int)(hostid >> 8),
             (short int)(hostid >> 16), (short int)(hostid >> 24));

   return strdup(uuid);
}

