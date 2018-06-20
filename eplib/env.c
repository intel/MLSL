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
#include "common.h"
#include "debug.h"
#include "env.h"

#ifdef ENABLE_CLIENT_ONLY
#include <mpi.h>
#endif
#include <string.h> /* memset */

int         max_ep        = MAX_EP_DEFAULT;      /* number of endpoints pe process */ /* TODO: rename to ep_count */
int         ep_per_node   = MAX_EP_DEFAULT;      /* number of endpoints per node */
int         ppn           = 1;                   /* number of masters per node */
size_t      shm_size      = SHM_SIZE_DEFAULT;    /* shm-region size in bytes */
int         check_mem_size = 0;                  /* whether to check size of allocated memory */
int         use_mem_hooks = 0;                   /* whether to enable memory hooks */
int         set_mem_hooks = 0;                   /* whether to set memory hooks */
int         server_affinity[AFFINITY_LEN] = {0}; /* per server affinity array */
int std_mpi_mode = STD_MPI_MODE_NONE;
int std_mpi_mode_implicit_allreduce_threshold = 1024;
int std_mpi_mode_implicit_alltoall_threshold = 1024;
int dynamic_server = DYNAMIC_SERVER_HYBRID;      /* dynamic server thread|process|hybrid|disable */
int         msg_priority           = 0;
size_t      msg_priority_threshold = MSG_PRIORITY_THRESHOLD_DEFAULT;
int         msg_priority_mode      = 1;

#ifdef ENABLE_CLIENT_ONLY
int use_allocator = 0;                           /* whether to use internal allocator */
#else
int use_allocator = 1;                           /* whether to use internal allocator */
#endif

size_t thp_threshold_mb = THP_THRESHOLD_MB_DEFAULT; /* Transparent Huge Pages threshold in MB */

/* string copies of env parameters to pass them over spawn API to servers */
char max_ep_str[MAX_EP_STR_LEN]                   = {0};
char ep_per_node_str[MAX_EP_STR_LEN]              = {0};
char shm_size_str[SHM_SIZE_STR_LEN]               = {0};
char server_affinity_str[SERVER_AFFINITY_STR_LEN] = {0};
char uuid_str[UUID_STR_LEN]                       = {0};
char eplib_root[EPLIB_ROOT_STR_LEN]               = {0};
char server_path[SERVER_PATH_STR_LEN]             = {0};
char dynamic_server_str[DYNAMIC_SERVER_STR_LEN]   = {0};
char msg_priority_str[MSG_PRIORITY_STR_LEN]       = {0};
char msg_priority_threshold_str[MSG_PRIORITY_THRESHOLD_STR_LEN] = {0};
char msg_priority_mode_str[MSG_PRIORITY_MODE_STR_LEN] = {0};

void parse_dynamic_server(const char* dynamic_server_to_parse)
{
    DEBUG_PRINT("dynamic_server_to_parse: %s\n", dynamic_server_to_parse);
    if (dynamic_server_to_parse != NULL)
    {
        if (strncmp(dynamic_server_to_parse, "hybrid", DYNAMIC_SERVER_STR_LEN) == 0)
            dynamic_server = DYNAMIC_SERVER_HYBRID;
        else if (strncmp(dynamic_server_to_parse, "process", DYNAMIC_SERVER_STR_LEN) == 0)
            dynamic_server = DYNAMIC_SERVER_PROCESS;
        else if (strncmp(dynamic_server_to_parse, "thread", DYNAMIC_SERVER_STR_LEN) == 0)
            dynamic_server = DYNAMIC_SERVER_THREAD;
        else if (strncmp(dynamic_server_to_parse, "asyncthread", DYNAMIC_SERVER_STR_LEN) == 0)
            dynamic_server = DYNAMIC_SERVER_ASYNCTHREAD;
        else if (strncmp(dynamic_server_to_parse, "disable", DYNAMIC_SERVER_STR_LEN) == 0)
            dynamic_server = DYNAMIC_SERVER_DISABLE;
        snprintf(dynamic_server_str, DYNAMIC_SERVER_STR_LEN, "%s", dynamic_server_to_parse);
    }

    DEBUG_PRINT("dynamic_server %s %d\n", dynamic_server_str, dynamic_server);
}

void parse_max_ep(const char* max_ep_to_parse)
{
    DEBUG_PRINT("max_ep_to_parse: %s\n", max_ep_to_parse);
    if (max_ep_to_parse != NULL)
    {
        max_ep = atoi(max_ep_to_parse);
    }
    ASSERT((max_ep >= 0 && max_ep <= MAX_EP_LIMIT));
    snprintf(max_ep_str, MAX_EP_STR_LEN, "%d", max_ep);

    if (max_ep == 1)
    {
        if (dynamic_server == DYNAMIC_SERVER_HYBRID) dynamic_server = DYNAMIC_SERVER_THREAD;
    }
    else if (max_ep > 1)
    {
        if (dynamic_server == DYNAMIC_SERVER_HYBRID || dynamic_server == DYNAMIC_SERVER_PROCESS)
            dynamic_server = DYNAMIC_SERVER_PROCESS;
        else if (dynamic_server != DYNAMIC_SERVER_DISABLE)
            ERROR("EPLIB_MAX_EP_PER_TASK > 1 requires EPLIB_DYNAMIC_SERVER=hybrid|process|disable\n");
    }

    DEBUG_PRINT("max_ep %d\n", max_ep);
}

void parse_shm_size(const char* shm_size_to_parse)
{
    DEBUG_PRINT("shm_size_to_parse: %s\n", shm_size_to_parse);
    if (shm_size_to_parse != NULL)
        shm_size = atoi(shm_size_to_parse) * GIGABYTE;

    long total_size = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
    if (shm_size >= total_size)
        shm_size = total_size/2;

    size_t offset = (((sizeof(intptr_t*) * max_ep) / PAGE_SIZE) + 1) * PAGE_SIZE;
    shm_size += offset;

    ASSERT(shm_size > 0);
    snprintf(shm_size_str, SHM_SIZE_STR_LEN, "%zu", shm_size / GIGABYTE);
    DEBUG_PRINT("shm_size %ld, total_size %ld\n", shm_size, total_size);
}

void parse_check_mem_size(const char* check_mem_size_to_parse)
{
    DEBUG_PRINT("check_mem_size_to_parse: %s\n", check_mem_size_to_parse);
    if (check_mem_size_to_parse != NULL)
        check_mem_size = atoi(check_mem_size_to_parse);
    ASSERT(check_mem_size == 0 || check_mem_size == 1);
    DEBUG_PRINT("check_mem_size %d\n", check_mem_size);
}

void parse_msg_priority(const char* msg_priority_to_parse)
{
    DEBUG_PRINT("msg_priority_to_parse: %s\n", msg_priority_to_parse);
    if (msg_priority_to_parse != NULL)
        msg_priority = atoi(msg_priority_to_parse);
    ASSERT(msg_priority == 0 || msg_priority == 1);
    snprintf(msg_priority_str, MSG_PRIORITY_STR_LEN, "%d", msg_priority);
    DEBUG_PRINT("msg_priority %d\n", msg_priority);
}

void parse_msg_priority_threshold(const char* msg_priority_threshold_to_parse)
{
    DEBUG_PRINT("msg_priority_threshold_to_parse: %s\n", msg_priority_threshold_to_parse);
    if (msg_priority_threshold_to_parse != NULL)
    {
        msg_priority_threshold = atoi(msg_priority_threshold_to_parse);
        ASSERT(msg_priority_threshold > 0);
    }
    snprintf(msg_priority_threshold_str, MSG_PRIORITY_THRESHOLD_STR_LEN, "%zu", msg_priority_threshold);
    DEBUG_PRINT("msg_priority_threshold %zu\n", msg_priority_threshold);
}

void parse_msg_priority_mode(const char* msg_priority_mode_to_parse)
{
    DEBUG_PRINT("msg_priority_mode_to_parse: %s\n", msg_priority_mode_to_parse);
    if (msg_priority_mode_to_parse != NULL)
        msg_priority_mode = atoi(msg_priority_mode_to_parse);
    ASSERT(msg_priority_mode == 0 || msg_priority_mode == 1);
    snprintf(msg_priority_mode_str, MSG_PRIORITY_MODE_STR_LEN, "%d", msg_priority_mode);
    DEBUG_PRINT("msg_priority_mode %d\n", msg_priority_mode);
}

void parse_server_affinity(const char* server_affinity_to_parse)
{
    int ep_idx = 0;
    size_t parse_count = 0;
    if (server_affinity_to_parse != NULL)
    {
        ASSERT(strlen(server_affinity_to_parse) > 0);

        /* need to create copy of original buffer cause it can be modified in strsep */
        size_t bytes_to_copy = sizeof(char) * (strlen(server_affinity_to_parse) + 1);
        char* server_affinity_copy = malloc(bytes_to_copy);
        ASSERT(server_affinity_copy);
        snprintf(server_affinity_copy, bytes_to_copy, "%s", server_affinity_to_parse);

        char* tmp = (char*)server_affinity_copy;
        for (ep_idx = 0; ep_idx < AFFINITY_LEN; ep_idx++)
        {
            char* affinity_str = strsep(&tmp, " \n\t,");
            if (affinity_str != NULL)
            {
                server_affinity[ep_idx] = atoi(affinity_str);
                ASSERT(server_affinity[ep_idx] >= 0);
                parse_count++;
            }
#ifdef ENABLE_CLIENT_ONLY
            else
            {
                ASSERT(ep_idx != 0);
                server_affinity[ep_idx] = server_affinity[ep_idx % parse_count];
            }
#endif
        }
        free(server_affinity_copy);
        if (parse_count < max_ep)
        {
            PRINT("number of comm cores (%zu) is less than number of eplib servers (%d)\n",
                  parse_count, max_ep);
            ASSERT(0);
        }
    }
    else
    {
        parse_count = AFFINITY_LEN;
        size_t core_count = sysconf(_SC_NPROCESSORS_ONLN);
        for (ep_idx = 0; ep_idx < AFFINITY_LEN; ep_idx++)
        {
            if (ep_idx < core_count)
                server_affinity[ep_idx] = core_count - ep_idx - 1;
            else
                server_affinity[ep_idx] = server_affinity[ep_idx % core_count];
        }
    }

#ifdef ENABLE_CLIENT_ONLY
    ep_per_node = AFFINITY_LEN;
#else
    ep_per_node = parse_count;
#endif
    for (ep_idx = 0; ep_idx < ep_per_node; ep_idx++)
    {
        ASSERT(server_affinity[ep_idx] >= 0);
        char core_str[CORE_STR_LEN] = {0};
        snprintf(core_str, CORE_STR_LEN, "%d", server_affinity[ep_idx]);
        strncat(server_affinity_str, core_str, CORE_STR_LEN);
        if (ep_idx != (ep_per_node - 1))
            strncat(server_affinity_str, ",", 1);
    }
    DEBUG_PRINT("server_affinity %s\n", server_affinity_str);
}

#ifdef ENABLE_CLIENT_ONLY
void modify_server_affinity()
{
    DEBUG_PRINT("modify_server_affinity: affinity %s, ppn %d, max_ep %d\n", server_affinity_str, ppn, max_ep);

    ep_per_node = ppn * max_ep;
    snprintf(ep_per_node_str, MAX_EP_STR_LEN, "%d", ep_per_node);

    memset(server_affinity + ep_per_node, 0, (AFFINITY_LEN - ep_per_node) * sizeof(int));
    memset(server_affinity_str, 0, AFFINITY_LEN);

    size_t ep_idx;
    ASSERT(ep_per_node > 0);
    for (ep_idx = 0; ep_idx < ep_per_node; ep_idx++)
    {
        ASSERT(server_affinity[ep_idx] >= 0);
        char core_str[CORE_STR_LEN] = {0};
        snprintf(core_str, CORE_STR_LEN, "%d", server_affinity[ep_idx]);
        strncat(server_affinity_str, core_str, CORE_STR_LEN);
        if (ep_idx != (ep_per_node - 1))
            strncat(server_affinity_str, ",", 1);
    }
    setenv("EPLIB_SERVER_AFFINITY", server_affinity_str, 1);
    DEBUG_PRINT("modified server_affinity %s\n", server_affinity_str);
}
#endif

void parse_uuid(const char* uuid_to_parse)
{
    DEBUG_PRINT("uuid_to_parse: %s\n", uuid_to_parse);
    if (uuid_to_parse != NULL)
        snprintf(uuid_str, UUID_STR_LEN, "%s", uuid_to_parse);
#ifdef ENABLE_CLIENT_ONLY
    else
    {
        char* uuid_tmp = get_uuid();
        ASSERT(uuid_tmp);
        snprintf(uuid_str, UUID_STR_LEN, "%s", uuid_tmp);
        free(uuid_tmp);
    }
    PMPI_Bcast(uuid_str, UUID_STR_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    DEBUG_PRINT("uuid %s\n", uuid_str);
}

void set_local_uuid(int local_id)
{
    /* If server launched as separate job, user expected to set EPLIB_UUID */
    if (dynamic_server == DYNAMIC_SERVER_DISABLE) return;

    ASSERT(strlen(uuid_str) == strlen(UUID_DEFAULT));

    size_t offset = UUID_STR_LEN - LOCAL_ID_STR_LEN;
    snprintf(uuid_str + offset, LOCAL_ID_STR_LEN, "%04d", local_id);
    DEBUG_PRINT("local_id %d, local_uuid %s\n", local_id, uuid_str);
}

void parse_thp_threshold(const char* thp_threshold_to_parse)
{
    DEBUG_PRINT("thp_threshold_to_parse: %s\n", thp_threshold_to_parse);
    if (thp_threshold_to_parse != NULL)
        thp_threshold_mb = atoi(thp_threshold_to_parse);
    DEBUG_PRINT("thp_threshold_mb %zu\n", thp_threshold_mb);
}

#ifdef ENABLE_CLIENT_ONLY

void parse_ppn()
{
    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);
    ASSERT(is_mpi_inited);

    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm node_comm;
    PMPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    PMPI_Comm_size(node_comm, &ppn);
    PMPI_Comm_free(&node_comm);

    DEBUG_PRINT("ppn %d\n", ppn);
}

void parse_use_allocator(const char* use_allocator_to_parse)
{
    DEBUG_PRINT("use_allocator_to_parse: %s\n", use_allocator_to_parse);
    if (use_allocator_to_parse != NULL)
        use_allocator = atoi(use_allocator_to_parse);
    ASSERT(use_allocator == 0 || use_allocator == 1);
    DEBUG_PRINT("use_allocator %d\n", use_allocator);
}

void parse_use_mem_hooks(const char* use_mem_hooks_to_parse)
{
    DEBUG_PRINT("use_mem_hooks_to_parse: %s\n", use_mem_hooks_to_parse);
    if (use_mem_hooks_to_parse != NULL)
        use_mem_hooks = atoi(use_mem_hooks_to_parse);
    ASSERT(use_mem_hooks == 0 || use_mem_hooks == 1);
    DEBUG_PRINT("use_mem_hooks %d\n", use_mem_hooks);
}

void parse_eplib_root(const char* eplib_root_to_parse)
{
    DEBUG_PRINT("eplib_root_to_parse: %s\n", eplib_root_to_parse);
    ASSERT_FMT(eplib_root_to_parse, "set EPLIB_ROOT");
    snprintf(eplib_root, EPLIB_ROOT_STR_LEN, "%s", eplib_root_to_parse);
    snprintf(server_path, SERVER_PATH_STR_LEN, "%s%s", eplib_root, SERVER_PATH_SUFFIX);
    DEBUG_PRINT("eplib_root %s, server_path %s\n", eplib_root, server_path);
}

void parse_std_mpi_mode(const char* std_mpi_mode_to_parse)
{
    DEBUG_PRINT("std_mpi_mode_to_parse: %s\n", std_mpi_mode_to_parse);
    if (std_mpi_mode_to_parse != NULL)
    {
        if (strncmp(std_mpi_mode_to_parse, "implicit", STD_MPI_MODE_STR_LEN) == 0)
        {
            std_mpi_mode = STD_MPI_MODE_IMPLICIT;
            char* allreduce_threshold_to_parse = getenv("EPLIB_STD_MPI_MODE_IMPLICIT_ALLREDUCE_THRESHOLD");
            if (allreduce_threshold_to_parse != NULL)
                std_mpi_mode_implicit_allreduce_threshold = atoi(allreduce_threshold_to_parse);

            char* alltoall_threshold_to_parse = getenv("EPLIB_STD_MPI_MODE_IMPLICIT_ALLTOALL_THRESHOLD");
            if (alltoall_threshold_to_parse != NULL)
                std_mpi_mode_implicit_alltoall_threshold = atoi(alltoall_threshold_to_parse);
        }
        else if (strncmp(std_mpi_mode_to_parse, "explicit", STD_MPI_MODE_STR_LEN) == 0)
            std_mpi_mode = STD_MPI_MODE_EXPLICIT;
        else
            std_mpi_mode = STD_MPI_MODE_NONE;
    }
    DEBUG_PRINT("std_mpi_mode %d %s\n", std_mpi_mode, std_mpi_mode_to_parse);
}

#endif /* ENABLE_CLIENT_ONLY */

void process_env_vars()
{
    parse_dynamic_server(getenv("EPLIB_DYNAMIC_SERVER"));
    parse_max_ep(getenv("EPLIB_MAX_EP_PER_TASK"));
    parse_uuid(getenv("EPLIB_UUID"));
    if (max_ep == 0) return;
    parse_shm_size(getenv("EPLIB_SHM_SIZE_GB"));
    parse_server_affinity(getenv("EPLIB_SERVER_AFFINITY"));
    parse_thp_threshold(getenv("EPLIB_THP_THRESHOLD_MB"));
    parse_msg_priority(getenv("EPLIB_MSG_PRIORITY"));
    parse_msg_priority_threshold(getenv("EPLIB_MSG_PRIORITY_THRESHOLD"));
    parse_msg_priority_mode(getenv("EPLIB_MSG_PRIORITY_MODE"));

#ifdef ENABLE_CLIENT_ONLY
    parse_ppn();
    parse_use_allocator(getenv("EPLIB_USE_ALLOCATOR"));
    if (max_ep > 0 && !IS_DYNAMIC_SERVER_THREAD())
    {
        DEBUG_PRINT("set use_allocator 1\n");
        use_allocator = 1;
    }

    parse_use_mem_hooks(getenv("EPLIB_USE_MEM_HOOKS"));
    if (!use_allocator)
    {
        DEBUG_PRINT("set use_mem_hooks 0\n");
        use_mem_hooks = 0;
    }

    parse_check_mem_size(getenv("EPLIB_CHECK_MEM_SIZE"));
    parse_eplib_root(getenv("EPLIB_ROOT"));
    parse_std_mpi_mode(getenv("EPLIB_STD_MPI_MODE"));
    modify_server_affinity();
#endif
}
