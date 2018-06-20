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
#ifndef _CLIENT_H_
#define _CLIENT_H_

#include <mpi.h>

#include "common.h"
#include "quant.h"

struct __cqueue_t;

struct __client_t
{
    int clientid;               /* Unique id of client */
    int taskid;                 /* Rank of application process */
    struct __cqueue_t* cqueue;  /* Pointer to command queue */
} __attribute__ ((aligned (CACHELINE_SIZE)));

typedef struct __client_t client_t;

void client_init(int, int);
struct __cqueue_t* client_get_cqueue(int);
void client_multiplex_endpoints(int, int, int, int* , MPI_Comm []);
void client_finalize();
quant_params_t* quant_params_submit(quant_params_t* global_param);

void allocator_init();
void allocator_pre_destroy();
void allocator_destroy();

#endif /* _CLIENT_H_ */
