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
#ifndef _COMMON_H_
#define _COMMON_H_

#include <arpa/inet.h>
#include <assert.h>

#if ENABLE_CHKP
#include <chkp.h>
#endif

#include <fcntl.h>
#include <mm_malloc.h>
#include <mpi.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <unistd.h>
#include <xmmintrin.h>

#include "debug.h"
#include "env.h"

/* General */
#define PAGE_SIZE      4096
#define CACHELINE_SIZE 64
#define EP_SUCCESS     1
#define EP_FAILURE     0
#define ONE_MB         1048576
#define TWO_MB         2097152

#define STRINGIFY2(x) #x
#define STRINGIFY(x)  STRINGIFY2(x)

/* Malloc */
#ifdef __INTEL_COMPILER
#define MALLOC_ALIGN_IMPL(ptr, size, align) \
  do {                                      \
      ptr = _mm_malloc(size, align);        \
      ASSERT(ptr);                          \
  } while (0)
#define FREE_ALIGN_IMPL(ptr) _mm_free(ptr)
#else
#define MALLOC_ALIGN_IMPL(ptr, size, align)                     \
  do {                                                          \
      int pm_ret = posix_memalign((void**)(&ptr), align, size); \
      ASSERT((pm_ret == 0) && ptr);                             \
  } while (0)
#define FREE_ALIGN_IMPL(ptr) free(ptr)
#endif

#define MALLOC_ALIGN_WRAPPER(ptr, size, user_align)                     \
  do {                                                                  \
      size_t align = user_align;                                        \
      if ((align < CACHELINE_SIZE) && (CACHELINE_SIZE % align == 0))    \
        align = CACHELINE_SIZE;                                         \
      if ((size >= thp_threshold_mb * ONE_MB) && (TWO_MB % align == 0)) \
      {                                                                 \
          /* use Transparent Huge Pages */                              \
          DEBUG_PRINT("use THP, size %zu\n", size);                     \
          MALLOC_ALIGN_IMPL(ptr, size, TWO_MB);                         \
      }                                                                 \
      else                                                              \
      {                                                                 \
          MALLOC_ALIGN_IMPL(ptr, size, align);                          \
      }                                                                 \
  } while (0)

#if ENABLE_CHKP
#define MAKE_BOUNDS(ptr, size)         { ptr = __chkp_make_bounds(ptr, size); }
#define REMOVE_BOUNDS(ptr)             { ptr = __chkp_kill_bounds(ptr); } 
#define MALLOC_ALIGN(ptr, size, align) ({ MALLOC_ALIGN_WRAPPER(ptr, size, align); MAKE_BOUNDS(ptr, size); })
#else
#define MAKE_BOUNDS(ptr, size)
#define REMOVE_BOUNDS(ptr)
#define MALLOC_ALIGN(ptr, size, align) MALLOC_ALIGN_WRAPPER(ptr, size, align)
#endif

#define FREE_ALIGN(ptr) FREE_ALIGN_IMPL(ptr)

#endif /* _COMMON_H_ */
