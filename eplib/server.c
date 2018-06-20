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
 * MPI Endpoints Server Proxy
 */

#include <ctype.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "common.h"
#include "cqueue.h"
#include "debug.h"
#include "env.h"
#include "memory.h"
#include "server.h"
#include "sig_handler.h"
#include "tune.h"
#include "quant.h"

typedef enum server_creation_type
{
    sct_spawn  = 0,
    sct_mpirun = 1
} server_creation_type;

typedef enum server_hostname_type
{
    sht_mpi  = 0,
    sht_name = 1,
    sht_ip   = 2,
} server_hostname_type;

typedef struct
{
    pthread_t ts;
    int global_server_idx;   /* MPI rank of server */
    int global_server_count; /* Total number of proxy servers */
    cqueue_t* cqueue;        /* Pointer to command queue */
    MPI_Comm task_comm;      /* Endpoint ranks grouped by task */
} server_t;

server_t* my_server = NULL;

static void server_affinity_set(server_t* server)
{
    /* Set affinity */
    cpu_set_t cpuset;
    pthread_t current_thread = pthread_self();
    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(server_affinity[server->global_server_idx % ep_per_node], sizeof(cpu_set_t), &cpuset);

    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
        perror("setaffinity failed\n");

    /* Check if we set the affinity correctly */
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
        perror("sched_getaffinity failed\n");

    for (int core_idx = 0; core_idx < __CPU_SETSIZE; core_idx++)
        if (__CPU_ISSET_S(core_idx, sizeof(cpu_set_t), &cpuset))
            DEBUG_PRINT("[%d:%d] SERVER affinity %d\n", server->global_server_idx, server->global_server_count, core_idx);
}

#ifndef ENABLE_CLIENT_ONLY

static void server_finalize();

static void server_init(int argc, char* argv[])
{
    /* Process arguments or environment variables */
    if (argc == NUM_ARGS_TO_SERVER)
    {
        parse_dynamic_server(argv[1]);
        parse_max_ep(argv[2]);
        parse_shm_size(argv[3]);
        parse_server_affinity(argv[4]);
        parse_uuid(argv[5]);
        parse_msg_priority(argv[6]);
        parse_msg_priority_threshold(argv[7]);
        parse_msg_priority_mode(argv[8]);
    }
    else
        process_env_vars();

    ASSERT(max_ep > 0);

    init_sig_handlers();

    /* Set the default values for transport libs */
    tune();

    MALLOC_ALIGN(my_server, sizeof(server_t), CACHELINE_SIZE);

    /* Initialize MPI */

    if (getenv("EPLIB_MPI_THREAD_MULTIPLE"))
    {
        int provided;
        PMPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        ASSERT(provided == MPI_THREAD_MULTIPLE);
    }
    else
        PMPI_Init(&argc, &argv);

    PMPI_Comm_rank(MPI_COMM_WORLD, &my_server->global_server_idx);
    PMPI_Comm_size(MPI_COMM_WORLD, &my_server->global_server_count);

    int local_master_idx = my_server->global_server_idx / max_ep;
    int local_server_idx = my_server->global_server_idx % max_ep;

    set_local_uuid(local_master_idx);

    /* Create communicator for servers tied to a given task */
    PMPI_Comm_split(MPI_COMM_WORLD, local_master_idx, local_server_idx, &my_server->task_comm);

    DEBUG_PRINT("server parameters: max_ep %s, shm_size %s, server_affinity [%s], uuid %s\n",
                 max_ep_str, shm_size_str, server_affinity_str, uuid_str);

    /* Set server affinity */
    server_affinity_set(my_server);

    /* Initialize inter-process shmem manager */
    memory_init();

    /* Register default shmem region */
    int mem_id = server_memory_register(NULL /* allocate new */, shm_size, my_server->task_comm);
    if (mem_id == -1)
    {
        PRINT("Server unable to create default shared memory region\n");
        server_finalize();
    }

    /* Get cqueue address */
    my_server->cqueue = (cqueue_t*)memory_get_cqueue(local_server_idx);
}

static void server_finalize()
{
    ASSERT(max_ep > 0);

    /* Terminate / cleanup shmem manager */
    memory_finalize();

    if (my_server != NULL && my_server->task_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&my_server->task_comm);

    MPI_Comm parent_comm;
    PMPI_Comm_get_parent(&parent_comm);
    if (parent_comm != MPI_COMM_NULL)
        PMPI_Comm_free(&parent_comm);
    /* Terminate / cleanup MPI */
    PMPI_Finalize();

    FREE_ALIGN(my_server);

    fini_sig_handlers();
}

int server_memory_register(void* base_addr, size_t mem_size, MPI_Comm task_comm)
{
    const int max_attempts = 128;
    int mem_id = -1;

    for (int attempt_idx = 0; attempt_idx < max_attempts; attempt_idx++)
    {
        int is_success = 0;
        int success_count = 0;

        mem_id = memory_register(base_addr, mem_size, uuid_str, 0 /* server */);

        if (mem_id != -1) is_success = 1;

        PMPI_Allreduce(&is_success, &success_count, 1, MPI_INT, MPI_SUM, task_comm);

        if (success_count == max_ep) break;

        if(mem_id != -1) memory_release(mem_id);
        mem_id = -1;
        sleep(2);
    }

    memory_get_client_shm_base(mem_id);
    return mem_id;
}

int main(int argc, char* argv[])
{
    server_init(argc, argv);

    /* Process command queue */
    cqueue_process(my_server->cqueue, my_server->task_comm);

    server_finalize();

    return 0;
}

#else /* !ENABLE_CLIENT_ONLY */

MPI_Comm intercomm = MPI_COMM_NULL;
pid_t pid = 0;

#define HOSTNAME_LEN     1024
#define PROC_STR_LEN     1024
#define LAUNCH_ARG_COUNT 1024

char hostname[HOSTNAME_LEN] = {0};

static void get_ip()
{
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char* iface_name_env = getenv("EPLIB_IFACE_NAME");
    char* iface_idx_env = getenv("EPLIB_IFACE_IDX");

    if (!((iface_idx_env && strlen(iface_idx_env)) || iface_name_env))
        ERROR("set EPLIB_IFACE_NAME or EPLIB_IFACE_IDX\n");

    ASSERT(getifaddrs(&ifaddr) != -1);

    if (iface_name_env)
    {
        DEBUG_PRINT("use iface type = %s\n", iface_name_env);
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
        {
            if (ifa->ifa_addr == NULL) continue;
            if (strncmp(ifa->ifa_name, iface_name_env, strlen(iface_name_env)) == 0)
            {
                family = ifa->ifa_addr->sa_family;
                if (family == AF_INET || family == AF_INET6) break;
            }
        }
        if (!ifa) PRINT("can't find interface with prefix %s\n", iface_name_env);
    }
    else if (iface_idx_env)
    {
        int iface_idx;
        int isIdx = 1;
        for (int i = 0; i < strlen(iface_idx_env); i++)
        {
            if (!isdigit(iface_idx_env[i]))
            {
                isIdx = 0;
                break;
            }
        }
        if (!isIdx) ERROR("incorrect EPLIB_IFACE_IDX = %s", iface_idx_env);

        iface_idx = atoi(iface_idx_env);
        DEBUG_PRINT("use iface_idx = %d\n", iface_idx);
        ifa = ifaddr;
        for (int idx = 0; idx <= iface_idx;)
        {
            ifa = ifa->ifa_next;
            if (ifa == NULL) ERROR("EPLIB_IFACE_IDX is too big: %d\n", iface_idx);

            family = ifa->ifa_addr->sa_family;
            if (family == AF_INET || family == AF_INET6) idx++;
        }
        if (!ifa) PRINT("can't find interface with index %s\n", iface_idx_env);
    }

    ASSERT(ifa);
    ASSERT(family == AF_INET || family == AF_INET6);

    int gai_result = getnameinfo(ifa->ifa_addr,
                                 (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6),
                                 hostname, HOSTNAME_LEN,
                                 NULL, 0, NI_NUMERICHOST);

    if (gai_result != 0) ERROR("getnameinfo error: %s\n", gai_strerror(gai_result));
    freeifaddrs(ifaddr);
}

static void get_hostname()
{
    int hostlen;
    struct addrinfo hints, *info, *p;
    int gai_result;
    char h_name[HOSTNAME_LEN] = {0};
    int name_is_found = 0;

    char* name_env = getenv("EPLIB_HOSTNAME");
    if (name_env)
    {
        /* we can set different EPLIB_HOSTNAME for different ranks over IMPI option "-gtool" */
        /* for example: mpirun ... -gtool "env EPLIB_HOSTNAME=name1:0; env EPLIB_HOSTNAME=name2:1" ... */
        DEBUG_PRINT("use name_env\n");
        ASSERT(strlen(name_env) < HOSTNAME_LEN);
        strncpy(hostname, name_env, strlen(name_env));
    }
    else
    {
        server_hostname_type type = sht_mpi;
        char* type_env = getenv("EPLIB_HOSTNAME_TYPE");
        if (type_env) type = atoi(type_env);
        DEBUG_PRINT("use type_env, type = %d\n", type);

        switch (type)
        {
            case sht_mpi:
                /* use MPI to get hostname */
                MPI_Get_processor_name(hostname, &hostlen);
                break;
            case sht_name:
                /* try to retrieve the full hostname by getaddrinfo */
                memset(&hints, 0, sizeof hints);
                hints.ai_family = AF_UNSPEC; /* either IPV4 or IPV6 */
                hints.ai_socktype = SOCK_STREAM;
                hints.ai_flags = AI_CANONNAME;

                gethostname(h_name, HOSTNAME_LEN - 1);
                if ((gai_result = getaddrinfo(h_name, NULL, &hints, &info)) != 0)
                    ERROR("getaddrinfo error: %s\n", gai_strerror(gai_result));

                for (p = info; p != NULL; p = p->ai_next)
                {
                    if (p->ai_canonname)
                    {
                        ASSERT(strlen(p->ai_canonname) < HOSTNAME_LEN);
                        PRINT("hostname: %s\n", p->ai_canonname);
                        strncpy(hostname, p->ai_canonname, strlen(p->ai_canonname));
                        name_is_found = 1;
                        break;
                    }
                }
                freeaddrinfo(info);
                if (!name_is_found) ASSERT(0);
                break;
            case sht_ip:
                get_ip();
                break;
            default:
                ASSERT(0);
        }
    }
    DEBUG_PRINT("get_hostname returns: %s\n", hostname);
}

static void print_launch_args(char** args)
{
    for (size_t idx = 0; idx < LAUNCH_ARG_COUNT; idx++)
    {
        if (!args[idx]) break;
        DEBUG_PRINT("arg[%zu]: %s\n", idx, args[idx]);
    }
}

static void server_create_processes()
{
    char* hostlist = NULL;
    int num_tasks;
    int server_count;

    server_creation_type creation_type = sct_spawn;
    char* creation_type_env = getenv("EPLIB_SERVER_CREATION_TYPE");
    if (creation_type_env) creation_type = atoi(creation_type_env);
    DEBUG_PRINT("use server_creation_type_env, server_creation_type = %d\n", creation_type);

    PMPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    server_count = num_tasks * max_ep;

    pid = 0;
    hostlist = (char*)calloc(num_tasks * HOSTNAME_LEN, sizeof(char));

    get_hostname();
    PMPI_Allgather(hostname, HOSTNAME_LEN, MPI_CHAR, hostlist, HOSTNAME_LEN, MPI_CHAR, MPI_COMM_WORLD);

    size_t arg_idx = 0;
    char* server_prefix_copy = NULL;
    char* server_prefix_env = getenv("EPLIB_SERVER_PREFIX");
    if (server_prefix_env)
    {
        size_t chars_to_copy = strlen(server_prefix_env) + 1;
        server_prefix_copy = calloc(chars_to_copy, sizeof(char));
        ASSERT(server_prefix_copy);
        snprintf(server_prefix_copy, chars_to_copy * sizeof(char), "%s", server_prefix_env);
    }

    switch (creation_type)
    {
        case sct_spawn:
        {
            char** commands;
            int* nservers;
            char*** args;
            MPI_Info* hostinfo;
            int* errcodes;
            commands = (char**)calloc(server_count, sizeof(char*));
            nservers = (int*)calloc(server_count, sizeof(int));
            args     = (char***)calloc(server_count, sizeof(char**));
            hostinfo = (MPI_Info*)calloc(server_count, sizeof(MPI_Info));
            errcodes = (int*)calloc(server_count, sizeof(int));
            ASSERT(commands && nservers && args && hostinfo && hostlist && errcodes);

            char* spawn_command = NULL;
            char* spawn_args[LAUNCH_ARG_COUNT];
            if (server_prefix_copy)
            {
                char* tmp = server_prefix_copy;
                char* prefix_part_str;
                while ((prefix_part_str = strsep(&tmp, " ,")))
                {
                    if (strlen(prefix_part_str))
                    {
                        if (!spawn_command) spawn_command = prefix_part_str;
                        else spawn_args[arg_idx++] = prefix_part_str;
                    }
                }
                spawn_args[arg_idx++] = server_path;
            }
            else
                spawn_command = server_path;

            size_t cur_arg_idx = arg_idx;
            spawn_args[arg_idx++] = dynamic_server_str;
            spawn_args[arg_idx++] = max_ep_str;
            spawn_args[arg_idx++] = shm_size_str;
            spawn_args[arg_idx++] = server_affinity_str;
            spawn_args[arg_idx++] = (char*)uuid_str;
            spawn_args[arg_idx++] = msg_priority_str;
            spawn_args[arg_idx++] = msg_priority_threshold_str;
            spawn_args[arg_idx++] = msg_priority_mode_str;
            spawn_args[arg_idx++] = (char*)0;
            ASSERT(arg_idx <= LAUNCH_ARG_COUNT);
            ASSERT((arg_idx - cur_arg_idx) == NUM_ARGS_TO_SERVER);

            DEBUG_PRINT("spawn_command: %s\n", spawn_command);
            print_launch_args(spawn_args);

            for (int i = 0; i < server_count; i++)
            {
                PMPI_Info_create(&hostinfo[i]);
                strncpy(hostname, &hostlist[(i / max_ep) * HOSTNAME_LEN], HOSTNAME_LEN - 1);
                hostname[HOSTNAME_LEN - 1] = '\0';
                PMPI_Info_set(hostinfo[i], "host", hostname);

                nservers[i] = 1;
                commands[i] = spawn_command;
                args[i] = spawn_args;
            }

            PMPI_Comm_spawn_multiple(server_count, commands, args, nservers, hostinfo, 0, MPI_COMM_WORLD, &intercomm, errcodes);

            for (int i = 0; i < server_count; i++)
                PMPI_Info_free(&hostinfo[i]);

            free(commands);
            free(nservers);
            free(args);
            free(hostinfo);
            free(errcodes);
            break;
        }
        case sct_mpirun:
        {
            int rank;
            PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                pid = fork();
                switch (pid)
                {
                    case -1:
                        ERROR("fork() returned -1\n");
                        break;
                    case 0:
                    {
                        /* inside forked process */

                        /*{ "mpirun",
                          "-localhost", "%s",
                          "-n", "%d",
                          "-ppn", "%d",
                          "-l",
                          "-hosts", "%s",
                          "[prefixes]",
                          "ep_server"
                        };*/
                        char* mpirun_args[LAUNCH_ARG_COUNT];

                        char* server_count_str = (char*)calloc(PROC_STR_LEN, sizeof(char));
                        char* ep_server_path = (char*)calloc((strlen(eplib_root) + strlen("/ep_server") + 1), sizeof(char));
                        char* localhost_str = (char*)calloc(HOSTNAME_LEN, sizeof(char));
                        ASSERT(server_count_str && ep_server_path && localhost_str);

                        strncpy(localhost_str, &hostlist[0], strlen(&hostlist[0]));
                        sprintf(ep_server_path, "%s/ep_server", eplib_root);
                        sprintf(server_count_str, "%d", server_count);

                        size_t host_count = server_count / ep_per_node;

                        /* for each hostname and for ',' as separators */
                        int hostlist_size = host_count * HOSTNAME_LEN + (host_count - 1);
                        char* hostlist_str = (char*)calloc(hostlist_size, sizeof(char));
                        int list_len = 0, idx = 0;
                        for (idx = 0; idx < host_count; idx++)
                        {
                            size_t chars_to_copy = strlen(&hostlist[idx * ppn * HOSTNAME_LEN]);
                            strncpy(&hostlist_str[list_len], &hostlist[idx * ppn * HOSTNAME_LEN], chars_to_copy);
                            list_len += chars_to_copy;

                            if (idx < (host_count - 1))
                                strncpy(&hostlist_str[list_len], ",", 1);
                            else
                                strncpy(&hostlist_str[list_len], "\0", 1);
                            list_len++;
                        }

                        setenv("EPLIB_UUID", uuid_str, 1);

                        /* EPLIB_* env variables are inherited from parent process and will be propagated to ep_server over mpirun */

                        mpirun_args[arg_idx++] = "mpiexec.hydra";
                        mpirun_args[arg_idx++] = "-localhost";
                        mpirun_args[arg_idx++] = localhost_str;
                        mpirun_args[arg_idx++] = "-n";
                        mpirun_args[arg_idx++] = server_count_str;
                        mpirun_args[arg_idx++] =  "-ppn";
                        mpirun_args[arg_idx++] = ep_per_node_str;
                        mpirun_args[arg_idx++] =  "-l";
                        mpirun_args[arg_idx++] ="-hosts";
                        mpirun_args[arg_idx++] = hostlist_str;

                        if (server_prefix_copy)
                        {
                            char* tmp = server_prefix_copy;
                            char* prefix_part_str;
                            while ((prefix_part_str = strsep(&tmp, " ,")))
                            {
                                if (strlen(prefix_part_str))
                                    mpirun_args[arg_idx++] = prefix_part_str;
                            }
                        }

                        mpirun_args[arg_idx++] = ep_server_path;
                        mpirun_args[arg_idx++] = (char*)0;
                        ASSERT(arg_idx <= LAUNCH_ARG_COUNT);
                        print_launch_args(mpirun_args);

                        execvp("mpiexec.hydra", mpirun_args);
                        
                        free(server_count_str);
                        free(ep_server_path);
                        free(localhost_str);
                        free(hostlist_str);

                        exit(0);
                    }
                    default:
                        DEBUG_PRINT("child pid: %d\n", pid);
                }
            }
            PMPI_Barrier(MPI_COMM_WORLD);
            break;
        }
        default:
            ASSERT(0);
    }

    if (server_prefix_copy) free(server_prefix_copy);
    free(hostlist);
}

static void* thread_main(void *arg)
{
    /* Set server affinity */
    server_affinity_set(my_server);

    /* Process command queue */
    cqueue_process(my_server->cqueue, my_server->task_comm);

    return 0;
}

static void server_create_thread()
{
    MALLOC_ALIGN(my_server, sizeof(server_t), CACHELINE_SIZE);
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_server->global_server_idx);
    PMPI_Comm_size(MPI_COMM_WORLD, &my_server->global_server_count);
    my_server->cqueue = cqueue_attach(0);
    my_server->task_comm = MPI_COMM_SELF;
    pthread_create(&my_server->ts, NULL, thread_main, (void *) my_server);
}

int server_memory_register(void* base_addr, size_t mem_size, MPI_Comm task_comm)
{
    return 0;
}

void server_create()
{
    if (dynamic_server == DYNAMIC_SERVER_PROCESS)
        server_create_processes();
    else if (IS_DYNAMIC_SERVER_THREAD())
        server_create_thread();
}

void server_destroy()
{
    if (IS_DYNAMIC_SERVER_THREAD())
    {
        pthread_join(my_server->ts, NULL);
        FREE_ALIGN(my_server);
    }

    if (intercomm != MPI_COMM_NULL)
    {
        PMPI_Comm_free(&intercomm);
        intercomm = MPI_COMM_NULL;
    }

    if (pid != 0)
    {
        int status;
        waitpid(pid, &status, 0);
    }
}

#endif /* !ENABLE_CLIENT_ONLY */
