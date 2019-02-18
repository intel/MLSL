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
#ifndef _ENV_H_
#define _ENV_H_

#include <stdlib.h>

#include "uuid.h"

#define MAX_EP_DEFAULT               0
#define MAX_EP_LIMIT                 16
#define MAX_EP_STR_LEN               4

#define MAX_PPN                      8
#define AFFINITY_LEN                 (MAX_PPN * MAX_EP_LIMIT)

#define SHM_SIZE_STR_LEN             12
#define GIGABYTE                     (1024L * 1024L * 1024L)
#define SHM_SIZE_DEFAULT             (4 * GIGABYTE)

#define THP_THRESHOLD_MB_DEFAULT     128

#define CORE_STR_LEN                 5 // idx of cpu core
#define SERVER_AFFINITY_STR_LEN      (CORE_STR_LEN * AFFINITY_LEN + AFFINITY_LEN) // to hold each core idxs and "," after each core

#define EPLIB_ROOT_STR_LEN           4096
#define SERVER_PATH_SUFFIX           "/ep_server"
#define SERVER_PATH_STR_LEN          (EPLIB_ROOT_STR_LEN + sizeof(SERVER_PATH_SUFFIX))

#define LOCAL_ID_STR_LEN             5

#define STD_MPI_MODE_STR_LEN         10
#define STD_MPI_MODE_NONE            0
#define STD_MPI_MODE_IMPLICIT        1
#define STD_MPI_MODE_EXPLICIT        2

#define DYNAMIC_SERVER_STR_LEN       12
#define DYNAMIC_SERVER_DISABLE       0
#define DYNAMIC_SERVER_THREAD        1
#define DYNAMIC_SERVER_ASYNCTHREAD   2
#define DYNAMIC_SERVER_PROCESS       3
#define DYNAMIC_SERVER_HYBRID        4

#define MSG_PRIORITY_STR_LEN           2
#define MSG_PRIORITY_THRESHOLD_STR_LEN 128
#define MSG_PRIORITY_THRESHOLD_DEFAULT 10000
#define MSG_PRIORITY_MODE_STR_LEN      2

#define NUM_ARGS_TO_SERVER           9  // (8 paramaters + NULL string) on client side, (binary_name + 8 parameters) on server side

#define IS_DYNAMIC_SERVER_THREAD()  ((dynamic_server == DYNAMIC_SERVER_ASYNCTHREAD || dynamic_server == DYNAMIC_SERVER_THREAD) ? 1 : 0)

extern int max_ep;                        /* number of endpoints pe process */
extern int ep_per_node;                   /* number of endpoints per node */
extern int ppn;                           /* number of masters per node */
extern size_t shm_size;                   /* shm-region size in gigabytes */
extern int check_mem_size;                /* whether to check size of allocated memory */
extern int use_mem_hooks;                 /* whether to enable memory hooks */
extern int set_mem_hooks;                 /* whether to set memory hooks */
extern int server_affinity[AFFINITY_LEN]; /* per server affinity array, on server side contains affinity cores for node's processes */
extern int dynamic_server;                /* dynamic server thread|process|hybrid|disable */
extern int use_allocator;                 /* whether to use internal allocator */
extern size_t thp_threshold_mb;           /* Transparent Huge Pages threshold in MB */
extern int msg_priority;                  /* whether to enable Rabenseifner’s implementation of Allreduce algorithm */
extern size_t msg_priority_threshold;     /* message size in bytes to enable Rabenseifner’s implementation of Allreduce algorithm */
extern int msg_priority_mode;             /* whether to enable Rabenseifner’s implementation of Allreduce algorithm with priority mode */

/* Baseline MPI acceleration options */
extern int std_mpi_mode;
extern int std_mpi_mode_implicit_allreduce_threshold;
extern int std_mpi_mode_implicit_alltoall_threshold;

extern char max_ep_str[MAX_EP_STR_LEN];
extern char ep_per_node_str[MAX_EP_STR_LEN];
extern char shm_size_str[SHM_SIZE_STR_LEN];
extern char server_affinity_str[SERVER_AFFINITY_STR_LEN];
extern char uuid_str[UUID_STR_LEN];
extern char eplib_root[EPLIB_ROOT_STR_LEN];
extern char server_path[SERVER_PATH_STR_LEN];
extern char dynamic_server_str[DYNAMIC_SERVER_STR_LEN];
extern char msg_priority_str[MSG_PRIORITY_STR_LEN];
extern char msg_priority_threshold_str[MSG_PRIORITY_THRESHOLD_STR_LEN];
extern char msg_priority_mode_str[MSG_PRIORITY_MODE_STR_LEN];

void process_env_vars();
void parse_max_ep(const char* max_ep_to_parse);
void parse_shm_size(const char* shm_size_to_parse);
void parse_check_mem_size(const char* check_mem_size_to_parse);
void parse_server_affinity(const char* affinity_to_parse);
void parse_uuid(const char* uuid_to_parse);
void set_local_uuid(int local_id);
void parse_dynamic_server(const char* dynamic_server_to_parse);
void parse_thp_threshold(const char* thp_threshold_to_parse);
void parse_msg_priority(const char* msg_priority_to_parse);
void parse_msg_priority_threshold(const char* msg_priority_threshold_to_parse);
void parse_msg_priority_mode(const char* msg_priority_mode_to_parse);

#ifdef ENABLE_CLIENT_ONLY
void parse_ppn();
void parse_use_allocator(const char* use_allocator_to_parse);
void parse_use_mem_hooks(const char* use_mem_hooks_to_parse);
void parse_eplib_root(const char* eplib_root_to_parse);
void parse_std_mpi_mode(const char* std_mpi_mode_to_parse);
void modify_server_affinity();
#endif

#endif /* _ENV_H */
