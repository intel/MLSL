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
#include <stdlib.h>

#include "client.h"
#include "common.h"
#include "cqueue.h"
#include "debug.h"
#include "env.h"
#include "handle.h"
#include "memory.h"
#include "request.h"
#include "sig_handler.h"
#include "window.h"
#include "eplib.h"

static int taskid = 0;
static int num_tasks = 1;
static int is_eplib_finalized = 0;

MPI_Request* block_coll_request;

void EPLIB_split_comm(MPI_Comm parentcomm, int color, int key, MPI_Comm newcomm)
{
    if (max_ep == 0) return;

    MPI_Comm* worldcomm = NULL, *peercomm = NULL;
    MALLOC_ALIGN(worldcomm, max_ep * sizeof(MPI_Comm), CACHELINE_SIZE);
    MALLOC_ALIGN(peercomm, max_ep * sizeof(MPI_Comm), CACHELINE_SIZE);

    for (int i = 0; i < max_ep; i++)
    {
        worldcomm[i] = MPI_COMM_NULL;
        peercomm[i] = MPI_COMM_NULL;
    }

    cqueue_create_server_comm(parentcomm, color, key, worldcomm, peercomm);
    handle_register(COMM_REG, newcomm, parentcomm, max_ep, 0,
                    num_tasks, taskid, worldcomm, peercomm);

    free(worldcomm);
    free(peercomm);
}

int EPLIB_comm_set_info(MPI_Comm* comms, size_t comm_count, const char* key, const char* value)
{
    size_t comm_idx;
    for (comm_idx = 0; comm_idx < comm_count; comm_idx++)
    {
        cqueue_t* mycqueue = handle_get_cqueue(comms[comm_idx]);
        cqueue_comm_set_info(mycqueue, comms[comm_idx], key, value);
    }
    return MPI_SUCCESS;
}

int EPLIB_init()
{
    process_env_vars();

    init_sig_handlers();

    PMPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    PMPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    set_local_uuid(taskid);

    allocator_init();

    if (max_ep == 0) return MPI_SUCCESS;

    /* Initialize client */
    client_init(taskid, num_tasks);

    /* Register MPI type and MPI Op before any other cqueue commands */
    cqueue_mpi_type_register();
    cqueue_mpi_op_register();

    /* Initialize communicator handles table */
    handle_init();

    /* Initialize window object table */
    window_init();

    /* Create server world and peer comm for MPI_COMM_WORLD */
    EPLIB_split_comm(MPI_COMM_WORLD, 0, taskid, MPI_COMM_WORLD);

    if (std_mpi_mode == STD_MPI_MODE_IMPLICIT)
        block_coll_request = malloc(max_ep*sizeof(MPI_Request));

    return MPI_SUCCESS;
}

int MPI_Init(int* argc, char*** argv)
{
    parse_dynamic_server(getenv("EPLIB_DYNAMIC_SERVER"));

    /* Initialize MPI */
    /* Special handling for async thread */
    int ret;
    if (dynamic_server == DYNAMIC_SERVER_ASYNCTHREAD)
    {
	int provided;
	ret = PMPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided != MPI_THREAD_MULTIPLE)
	{
	    PRINT("Requested thread level not provided.\n");
	    PMPI_Abort(MPI_COMM_WORLD, -1);
	}
    }
    else
	ret = PMPI_Init(argc, argv);

    /* Initialize EPLIB */
    EPLIB_init();

    return ret;
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    parse_dynamic_server(getenv("EPLIB_DYNAMIC_SERVER"));

    /* Initialize MPI */
    /* Special handling for async thread */
    int ret;
    if (dynamic_server == DYNAMIC_SERVER_ASYNCTHREAD)
    {
	ret = PMPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, provided);
	if (*provided != MPI_THREAD_MULTIPLE)
	{
	    PRINT("Requested thread level not provided.\n");
	    PMPI_Abort(MPI_COMM_WORLD, -1);
	}
    }
    else
    {
	ret = PMPI_Init_thread(argc, argv, required, provided);
	if (*provided != required)
	{
	    PRINT("Requested thread level not provided.\n");
	    PMPI_Abort(MPI_COMM_WORLD, -1);
	}
    }

    /* Initialize EPLIB */
    EPLIB_init();

    return ret;
}

int MPI_Comm_rank(MPI_Comm comm, int* rank_ptr)
{
    if (handle_get_type(comm) == COMM_EP)
    {
        *rank_ptr = handle_get_rank(comm);
        return MPI_SUCCESS;
    }
    else
        return PMPI_Comm_rank(comm, rank_ptr);
}

int MPI_Comm_size(MPI_Comm comm, int* size_ptr)
{
    if (handle_get_type(comm) == COMM_EP)
    {
        *size_ptr = handle_get_size(comm);
        return MPI_SUCCESS;
    }
    else
       return PMPI_Comm_size(comm, size_ptr);
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm)
{
    int ret;

    if (handle_get_type(comm) == COMM_EP)
       ERROR("MPI_Comm_split is not supported on endpoint communicator\n");
    else
    {
        ret = PMPI_Comm_split(comm, color, key, newcomm);
        EPLIB_split_comm(comm, color, key, *newcomm);
    }
    return ret;
}

int MPI_Comm_create_endpoints(MPI_Comm parent_comm, int num_endpoints, MPI_Info info, MPI_Comm out_comm_hdls[])
{
    int* num_ep_list;
    int tot_endpoints = 0;
    int local_epid, global_epid = 0;
    int i;

    /* Return error if requested endpoints greater than servers */
    if (num_endpoints > max_ep) return MPI_ERR_UNKNOWN;

    /* Map endpoints to servers */
    /* Step 1/3: Gather total number of endpoints requested by all
       application tasks in the parent communicator parent_comm */
    num_ep_list = (int*)malloc(num_tasks*sizeof(int));
    ASSERT(num_ep_list != NULL);
    PMPI_Allgather(&num_endpoints, 1, MPI_INT, (void*)num_ep_list, 1, MPI_INT, parent_comm);

    /* Step 2/3: Create communicator handle objects */
    for (i = 0; i < num_tasks; i++)
    {
        tot_endpoints += num_ep_list[i];
        if (i < taskid)
            global_epid += num_ep_list[i];
    }

    for (local_epid = 0; local_epid < num_endpoints; local_epid++)
    {
	out_comm_hdls[local_epid] = (MPI_Comm) handle_register(COMM_EP, MPI_COMM_NULL, parent_comm,
                                                    num_endpoints, local_epid, tot_endpoints,
                                                    global_epid, NULL, NULL);
        global_epid++;
    }

    /* Step 3/3: */
    client_multiplex_endpoints(max_ep, num_tasks, num_endpoints, num_ep_list, out_comm_hdls);

    PMPI_Barrier(parent_comm);

    free(num_ep_list);

    return MPI_SUCCESS;
}

int MPI_Comm_free(MPI_Comm* comm)
{
    int ret = MPI_SUCCESS;

    if (max_ep > 0)
    {
        int commtype = handle_get_type(*comm);
        handle_release(*comm);

        if (commtype == COMM_REG)
            ret = PMPI_Comm_free(comm);
    }
    else
        ret = PMPI_Comm_free(comm);

    return ret;
}

int MPI_Send(const void* buffer, int count, MPI_Datatype datatype,
             int dst, int tag, MPI_Comm comm)
{
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        return cqueue_send(mycqueue, buffer, count, datatype, dst, tag, comm);
    else
        return PMPI_Send(buffer, count, datatype, dst, tag, comm);
}

int MPI_Recv(void* buffer, int count, MPI_Datatype datatype,
             int src, int tag, MPI_Comm comm, MPI_Status* status)
{
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        return cqueue_recv(mycqueue, buffer, count, datatype, src, tag, comm, status);
    else
        return PMPI_Recv(buffer, count, datatype, src, tag, comm, status);
}

int MPI_Isend(const void* buffer, int count, MPI_Datatype datatype,
              int dst, int tag, MPI_Comm comm, MPI_Request* request)
{
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        return cqueue_isend(mycqueue, buffer, count, datatype, dst, tag, comm, request);
    else
    {
	if (std_mpi_mode == STD_MPI_MODE_IMPLICIT && max_ep > 0)
	    return cqueue_isend(client_get_cqueue((taskid + dst) % max_ep), buffer,
				count, datatype, dst, tag, comm, request);
        return PMPI_Isend(buffer, count, datatype, dst, tag, comm, request);
    }
}

int MPI_Irecv(void* buffer, int count, MPI_Datatype datatype,
              int src, int tag, MPI_Comm comm, MPI_Request* request)
{
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        return cqueue_irecv(mycqueue, buffer, count, datatype, src, tag, comm, request);
    else
    {
	if (std_mpi_mode == STD_MPI_MODE_IMPLICIT && max_ep > 0)
            return cqueue_irecv(client_get_cqueue((taskid + src) % max_ep), buffer,
                                count, datatype, src, tag, comm, request);
        return PMPI_Irecv(buffer, count, datatype, src, tag, comm, request);
    }
}

int MPI_Wait(MPI_Request* request, MPI_Status* status)
{
    if (*request != MPI_REQUEST_NULL)
    {
        centry_t* mycentry = cqueue_get_centry((long)*request);

        if (mycentry != NULL)
            return cqueue_wait(mycentry, request, status);
        else
            return PMPI_Wait(request, status);
    }

    /* Invalid MPI_Request. Either null or not a persistent request. */
    return MPI_ERR_REQUEST;
}

int MPI_Waitall(int count, MPI_Request* request, MPI_Status* status)
{
    int retall = MPI_SUCCESS;

    for(int i = 0; i < count; i++)
    {
        int ret = MPI_ERR_REQUEST;
        if (request[i] != MPI_REQUEST_NULL)
        {
            centry_t* mycentry = cqueue_get_centry((long)request[i]);
            if (mycentry != NULL)
            {
                if (status == MPI_STATUS_IGNORE)
                    ret = cqueue_wait(mycentry, &request[i], MPI_STATUS_IGNORE);
               else
                   ret = cqueue_wait(mycentry, &request[i], &status[i]);
            }
            else
            {
                if (status == MPI_STATUS_IGNORE)
                    ret = PMPI_Wait(&request[i], MPI_STATUS_IGNORE);
               else
                    ret = PMPI_Wait(&request[i], &status[i]);
            }
        }
        if (ret != MPI_SUCCESS) retall = ret;
    }
    return retall;
}

int MPI_Waitany(int count, MPI_Request* request, int* index, MPI_Status* status)
{
    while(1)
    {
        for (int i = 0; i < count; i++)
        {
            if (request[i] != MPI_REQUEST_NULL)
            {
                int flag, ret;
                centry_t* mycentry = cqueue_get_centry((long)request[i]);

                if (mycentry != NULL)
                    ret = cqueue_test(mycentry, &request[i], &flag, &status[i]);
                else
                    ret = PMPI_Test(&request[i], &flag, &status[i]);

                if (flag == 1)
                {
                    *index = i;
                    return ret;
                }
            }
        }
    }
    return MPI_SUCCESS;
}

int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status)
{
    if (*request != MPI_REQUEST_NULL)
    {
        centry_t* mycentry = cqueue_get_centry((long)*request);

        if (mycentry != NULL)
            return cqueue_test(mycentry, request, flag, status);
        else
            return PMPI_Test(request, flag, status);
    }
    return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_barrier(mycqueue, comm);
    else
        ret = PMPI_Barrier(comm);

    return ret;
}

#define GET_MESSAGE_PAYLOAD(id, num_ep, count, chunk, istart)        \
  do {                                                               \
      long iend;                                                     \
      long leftover = count % num_ep;                                \
      chunk = count / num_ep;                                        \
      if (id < leftover)                                             \
      {                                                              \
          istart = (chunk + 1) * id;                                 \
          iend = istart + chunk;                                     \
      }                                                              \
      else                                                           \
      {                                                              \
          istart = (chunk + 1) * leftover + chunk * (id - leftover); \
          iend = istart + chunk - 1;                                 \
      }                                                              \
      chunk = iend - istart + 1;                                     \
  } while (0)

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
    {
        MPI_Request tmprequest;
        cqueue_iallreduce(mycqueue, sendbuf, recvbuf, count, datatype, op, comm, &tmprequest);
        return MPI_Wait(&tmprequest, MPI_STATUS_IGNORE);
    }
    else
    {
	if (std_mpi_mode == STD_MPI_MODE_IMPLICIT && max_ep > 0)
        {
            int num_ep = 1;
            if (count >= std_mpi_mode_implicit_allreduce_threshold)
              num_ep = max_ep;

            for (int epid = 0; epid < num_ep; epid++)
            {
                long start, chunk;
                GET_MESSAGE_PAYLOAD(epid, num_ep, count, chunk, start);
                block_coll_request[epid] = MPI_REQUEST_NULL;
                ret = cqueue_iallreduce(client_get_cqueue(epid), (float*)sendbuf + start, (float*)recvbuf + start,
                                        chunk, datatype, op, comm, &block_coll_request[epid]);
            }
            return MPI_Waitall(num_ep, block_coll_request, MPI_STATUS_IGNORE);
        }
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    return ret;
}

int MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);
    if (mycqueue != NULL)
        ret = cqueue_iallreduce(mycqueue, sendbuf, recvbuf, count, datatype, op, comm, request);
    else
        ret = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);

    return ret;
}

int MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount,MPI_Datatype recvtype, MPI_Comm comm)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
    {
        MPI_Request tmprequest;
        cqueue_ialltoall(mycqueue, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, &tmprequest);
        return MPI_Wait(&tmprequest, MPI_STATUS_IGNORE);
    }
    else
    {
        if (std_mpi_mode == STD_MPI_MODE_IMPLICIT && max_ep > 0)
        {
            int num_ep = 1;
            if (sendcount >= std_mpi_mode_implicit_alltoall_threshold)
              num_ep = max_ep;

            for (int epid = 0; epid < num_ep; epid++)
            {
                int sendstart, sendchunk;
                int recvstart, recvchunk;
                GET_MESSAGE_PAYLOAD(epid, num_ep, sendcount, sendchunk, sendstart);
                GET_MESSAGE_PAYLOAD(epid, num_ep, recvcount, recvchunk, recvstart);
                block_coll_request[epid] = MPI_REQUEST_NULL;
                ret = cqueue_ialltoall(client_get_cqueue(epid), sendbuf + sendstart, sendchunk, sendtype,
                                       recvbuf+recvstart, recvchunk, recvtype, comm, &block_coll_request[epid]);
            }
            return MPI_Waitall(num_ep, block_coll_request, MPI_STATUS_IGNORE);
        }
        return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }

    return ret;
}

int MPI_Ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_ialltoall(mycqueue, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    else
        ret = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

    return ret;
}

int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
    {
        MPI_Request tmprequest;
        ret = cqueue_iallgather(mycqueue, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, &tmprequest);
        return MPI_Wait(&tmprequest, MPI_STATUS_IGNORE);
    }
    else
    {
        if (std_mpi_mode == STD_MPI_MODE_IMPLICIT && max_ep > 0)
        {
            int commsize;
            int typesize;
            MPI_Type_size(recvtype, &typesize);
            if (sendbuf == MPI_IN_PLACE)
            {
                PMPI_Comm_size(comm, &commsize);
                for (int rank = 0; rank < commsize; rank++)
                {
                    block_coll_request[rank] = MPI_REQUEST_NULL;
                    cqueue_ibcast(client_get_cqueue(rank % max_ep), recvbuf + rank * typesize * recvcount, recvcount,
                                  recvtype, rank, comm, &block_coll_request[rank]);
                }
                return MPI_Waitall(commsize, block_coll_request, MPI_STATUS_IGNORE);
            }
            else
              return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        }
        return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }

    return ret;
}

int MPI_Iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                   MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_iallgather(mycqueue, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    else
        ret = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

    return ret;
}

int MPI_Igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_igather(mycqueue, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
    else
        ret = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

    return ret;
}

int MPI_Ireduce_scatter_block(const void* sendbuf, void* recvbuf, int recvcount,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_ireduce_scatter_block(mycqueue, sendbuf, recvbuf, recvcount, datatype, op, comm, request);
    else
        ret = PMPI_Ireduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm, request);

    return ret;
}

int MPI_Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, int root, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_ireduce(mycqueue, sendbuf, recvbuf, count, datatype, op, root, comm, request);
    else
        ret = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);

    return ret;
}

int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
    {
        MPI_Request request;
        ret = cqueue_ireduce(mycqueue, sendbuf, recvbuf, count, datatype, op, root, comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else
        ret = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

    return ret;
}

int MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_ibcast(mycqueue, buffer, count, datatype, root, comm, request);
    else
        ret = PMPI_Ibcast(buffer, count, datatype, root, comm, request);

    return ret;
}

#ifdef ENABLE_MPIRMA_ENDPOINTS

int MPI_Win_allocate(MPI_Aint win_size, int disp_unit, MPI_Info info, MPI_Comm comm, void* baseptr, MPI_Win* winptr)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_win_allocate(mycqueue, win_size, disp_unit, info, comm, baseptr, winptr);
    else
        ret = PMPI_Win_allocate(win_size, disp_unit, info, comm, baseptr, winptr);

    /* Register window object */
    /* For EP-lib: value represent window object in server,
       replace with index of free entry in the window table.
       Additionally, MPI_Win_free should free the buffer */
    /* Default: track value in window table, don't modify */
    window_register(winptr, baseptr, myclient, 1);

    return ret;
}

int MPI_Win_create(void* base, MPI_Aint win_size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win* winptr)
{
    int ret;
    cqueue_t* mycqueue = handle_get_cqueue(comm);

    if (mycqueue != NULL)
        ret = cqueue_win_create(mycqueue, base, win_size, disp_unit, info, comm, winptr);
    else
        ret = PMPI_Win_create(base, win_size, disp_unit, info, comm, winptr);

    /* Register window object */
    /* For EP-lib: value represent window object in server,
       replace with index of free entry in the window table.
       Additionally, MPI_Win_free should *not* free the buffer */
    /* Default: track value in window table, don't modify */
    window_register(winptr, &base, myclient, 0);

    return ret;
}

int MPI_Win_free(MPI_Win* winptr)
{
    int ret;
    client_t* myclient = window_get_client(*winptr);
    cqueue_t* mycqueue = client_get_cqueue(myclient);
    MPI_Win win = *winptr;

    if (myclient == NULL)
        ret = PMPI_Win_free(winptr);
    else
        ret = cqueue_win_free(mycqueue, winptr);

    window_release(win);

    return ret;
}

int MPI_Put(const void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Put(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype, win);
    else
        ret = cqueue_put(mycqueue, origin_addr, origin_count, origin_datatype,
                         target_rank, target_disp, target_count, target_datatype, win);

    return ret;
}

int MPI_Get(void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Get(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype, win);
    else
        ret = cqueue_get(mycqueue, origin_addr, origin_count, origin_datatype,
                         target_rank, target_disp, target_count, target_datatype, win);

    return ret;
}

int MPI_Win_lock(int lock_type, int target_rank, int assert, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_lock(lock_type, target_rank, assert, win);
    else
        ret = cqueue_win_lock(mycqueue, lock_type, target_rank, assert, win);

    return ret;
}

int MPI_Win_unlock(int target_rank, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_unlock(target_rank, win);
    else
        ret = cqueue_win_unlock(mycqueue, target_rank, win);

    return ret;
}

int MPI_Win_lock_all(int assert, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_lock_all(assert, win);
    else
        ret = cqueue_win_lock_all(mycqueue, assert, win);

    return ret;
}

int MPI_Win_unlock_all(MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_unlock_all(win);
    else
        ret = cqueue_win_unlock_all(mycqueue, win);

    return ret;
}

int MPI_Win_flush(int target_rank, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_flush(target_rank, win);
    else
        ret = cqueue_win_flush(mycqueue, target_rank, win);

    return ret;
}

int MPI_Win_flush_local(int target_rank, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_flush_local(target_rank, win);
    else
        ret = cqueue_win_flush_local(mycqueue, target_rank, win);

    return ret;
}

int MPI_Win_flush_all(MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_flush_all(win);
    else
        ret = cqueue_win_flush_all(mycqueue, win);

    return ret;
}

int MPI_Win_fence(int assert, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Win_fence(assert, win);
    else
        ret = cqueue_win_fence(mycqueue, assert, win);

    return ret;
}

int MPI_Compare_and_swap(const void* origin_addr, const void* compare_addr, void* result_addr,
                         MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype,
                                    target_rank, target_disp, win);
    else
        ret = cqueue_compare_and_swap(mycqueue, origin_addr, compare_addr,
                                      result_addr, datatype, target_rank, target_disp, win);

    return ret;
}

int MPI_Fetch_and_op(const void* origin_addr, void* result_addr,
                     MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)
{
    int ret;
    client_t* myclient = window_get_client(win);
    cqueue_t* mycqueue = client_get_cqueue(myclient);

    if (myclient == NULL)
        ret = PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);
    else
        ret = cqueue_fetch_and_op(mycqueue, origin_addr, result_addr, datatype, target_rank, target_disp, op, win);

    return ret;
}

#endif /* !ENABLE_MPIRMA_ENDPOINTS */

int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void* baseptr)
{
    if (max_ep > 0)
    {
        *(char **)baseptr = EPLIB_memalign(PAGE_SIZE, size);
        if (*(char **)baseptr == NULL)
            return MPI_ERR_NO_MEM;
        return MPI_SUCCESS;
    }
    return PMPI_Alloc_mem(size, info, baseptr);
}

int MPI_Free_mem(void* baseptr)
{
    if (max_ep > 0)
    {
        EPLIB_free(baseptr);
        return MPI_SUCCESS;
    }
    return MPI_Free_mem(baseptr);
}

int MPI_Finalize()
{
    PMPI_Barrier(MPI_COMM_WORLD);
    EPLIB_finalize();
    PMPI_Barrier(MPI_COMM_WORLD);
    return PMPI_Finalize();
}

/*
 * EPLIB Interface
 */

int EPLIB_finalize()
{
    if (!__sync_bool_compare_and_swap(&is_eplib_finalized, 0, 1))
    {
        DEBUG_PRINT("already finalized, skip EPLIB_finalize\n");
        return MPI_SUCCESS;
    }

    allocator_pre_destroy();

    if (max_ep > 0)
    {
        if (std_mpi_mode == STD_MPI_MODE_IMPLICIT)
            free(block_coll_request);

        handle_finalize();
        window_finalize();
        client_finalize();
    }

    allocator_destroy();

    fini_sig_handlers();

    return MPI_SUCCESS;
}

void* EPLIB_malloc(size_t bytes)
{
    return memory_malloc(bytes);
}

void* EPLIB_realloc(void* ptr, size_t new_size)
{
    return memory_realloc(ptr, new_size);
}

void* EPLIB_calloc(size_t num, size_t elem_size)
{
    return memory_calloc(num, elem_size);
}

void* EPLIB_memalign(size_t alignment, size_t bytes)
{
    return memory_memalign(alignment, bytes);
}

void EPLIB_free(void* addr)
{
    memory_free(addr);
}

int EPLIB_memory_is_shmem(void* addr)
{
    if (max_ep == 0 || !addr || IS_DYNAMIC_SERVER_THREAD()) return 1;
    return memory_is_shmem(addr, NULL);
}

void EPLIB_set_mem_hooks()
{
    use_mem_hooks = 1;
}

void* EPLIB_quant_params_submit(void* global_param)
{
    return (void*)quant_params_submit((quant_params_t*)global_param);
}

void EPLIB_execute()
{
    cqueue_execute();
}

void EPLIB_suspend()
{
    cqueue_suspend();
}

FILE* EPLIB_fopen(int epid, const char* filename, const char* mode)
{
#ifdef ENABLE_FILEIO
    FILE* stream;
    cqueue_fopen(client_get_cqueue(epid), filename, mode, &stream);
    return stream;
#else
    return fopen(filename, mode);
#endif
}

size_t EPLIB_fread(int epid, void* buffer, size_t size, size_t count, FILE* stream)
{
#ifdef ENABLE_FILEIO
    size_t readsize;
    MPI_Request request;
    cqueue_fread_nb(client_get_cqueue(epid), buffer, size, count, stream, &request);
    fentry_t* myfentry = cqueue_get_fentry(request);
    cqueue_fwait(myfentry, &request, &readsize);
    return readsize;
#else
    return fread(buffer, size, count, stream);
#endif
}

size_t EPLIB_fread_nb(int epid, void* buffer, size_t size, size_t count, FILE* stream, MPI_Request* request)
{
#ifdef ENABLE_FILEIO
    cqueue_fread_nb(client_get_cqueue(epid), buffer, size, count, stream, request);
    return 0;
#else
    return fread(buffer, size, count, stream);
#endif
}

size_t EPLIB_forc_nb(int epid, const char* filename, const char* mode,
                     void* buffer, size_t size, size_t count, MPI_Request* request)
{
#ifdef ENABLE_FILEIO
    cqueue_forc_nb(client_get_cqueue(epid), filename, mode, buffer, size, count, request);
    return 0;
#else
    FILE* stream;
    size_t readsize = 0;
    stream = fopen(filename, mode);
    if (stream)
    {
        readsize = fread(buffer, size, count, stream);
        fclose(stream);
    }
    return readsize;
#endif
}

int EPLIB_fwait(MPI_Request* request, size_t* readcount)
{
#ifdef ENABLE_FILEIO
    if (*request != MPI_REQUEST_NULL)
    {
        fentry_t* myfentry = cqueue_get_fentry(*request);
        cqueue_fwait(myfentry, request, readcount);
    }
#endif
    return MPI_SUCCESS;
}

int EPLIB_fwaitall(int count, MPI_Request* request, size_t* readcount)
{
#ifdef ENABLE_FILEIO
    for(int i = 0; i < count; i++)
    {
        if (request[i] != MPI_REQUEST_NULL)
        {
            fentry_t* myfentry = cqueue_get_fentry(request[i]);
            cqueue_fwait(myfentry, &request[i], &readcount[i]);
        }
    }
#endif
    return MPI_SUCCESS;
}

int EPLIB_fclose(int epid, FILE* stream)
{
#ifdef ENABLE_FILEIO
    return cqueue_fclose(client_get_cqueue(epid), stream);
#else
    return fclose(stream);
#endif
}
