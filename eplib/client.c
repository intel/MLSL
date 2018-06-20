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
/*
 * MPI Endpoints Client Interface
 */

#include <signal.h>

#include "client.h"
#include "server.h"
#include "common.h"
#include "cqueue.h"
#include "debug.h"
#include "env.h"
#include "memory.h"

client_t* client_table;

void allocator_init()
{
    memory_get_limits();

    if (!use_allocator) return;

    /* Initialize and register inter-process shmem manager */
    int memid = -1;
    memory_init();
    memid = memory_register(NULL /* allocate new */, shm_size, uuid_str, 1 /* client */);
    if (memid == -1)
        ERROR("Client unable to create default shared memory region (%ld bytes)\n", shm_size);

    PMPI_Barrier(MPI_COMM_WORLD);
}

void allocator_pre_destroy()
{
    if (!use_allocator) return;
    memory_unlink();
}

void allocator_destroy()
{
    if (!use_allocator) return;
    if (!use_mem_hooks) memory_finalize();
}

void client_init(int taskid, int num_tasks)
{
    ASSERT(max_ep > 0);

    /* Create data structure to track all clients */
    MALLOC_ALIGN(client_table, sizeof(client_t) * max_ep, CACHELINE_SIZE);

    PMPI_Barrier(MPI_COMM_WORLD);

    /* Initialize command queues */
    cqueue_init(max_ep);

    /* Spawn server */
    server_create();

    /* Create client data-structure */
    for (int clientid = 0; clientid < max_ep; clientid++)
    {
        client_t* myclient = &client_table[clientid];
        myclient->clientid = clientid;
        myclient->taskid = taskid;
        /* Attach client to a command queue */
        myclient->cqueue = cqueue_attach(clientid);
        memory_set_cqueue(clientid, myclient->cqueue);
    }
    PMPI_Barrier(MPI_COMM_WORLD);

    if (use_mem_hooks) set_mem_hooks = 1;
}

inline struct __cqueue_t* client_get_cqueue(int clientid)
{
    return client_table[clientid].cqueue;
}

void client_multiplex_endpoints(int max_ep, int num_tasks, int num_endpoints, int* num_ept_list, MPI_Comm out_comm_hdls[])
{}

void client_finalize()
{
    ASSERT(max_ep > 0);
    cqueue_finalize();
    FREE_ALIGN(client_table);
    server_destroy();
}

#define MAX_FUNC_NAME_LEN 128
#define MAX_PATH_LEN 4096

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define STR_CPY(dst, src, max_len)                                  \
  do                                                                \
  {                                                                 \
          dst = memory_memalign(CACHELINE_SIZE, max_len);           \
          memset(dst, 0, max_len);                                  \
          strncpy(dst, src, MIN(max_len - 1, strlen(src)));         \
  }while (0)

quant_params_t* quant_params_submit(quant_params_t* global_params)
{
    ASSERT(global_params);
    quant_params_t* params = memory_memalign(PAGE_SIZE, sizeof(quant_params_t));

    ASSERT(global_params->lib_path);
    STR_CPY(params->lib_path,
            global_params->lib_path,
            MAX_PATH_LEN);

    ASSERT(global_params->quant_buffer_func_name);
    STR_CPY(params->quant_buffer_func_name,
            global_params->quant_buffer_func_name,
            MAX_FUNC_NAME_LEN);
    
    ASSERT(global_params->dequant_buffer_func_name);
    STR_CPY(params->dequant_buffer_func_name,
            global_params->dequant_buffer_func_name, 
            MAX_FUNC_NAME_LEN);

    ASSERT(global_params->reduce_sum_func_name);
    STR_CPY(params->reduce_sum_func_name,
            global_params->reduce_sum_func_name,
            MAX_FUNC_NAME_LEN);

    params->block_size = global_params->block_size;
    params->elem_in_block = global_params->elem_in_block;

    memory_set_quant_params(0, params);
    return params;
}

