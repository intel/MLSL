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
 * Circular queue implementation
 */

#include "allreduce_pr.h"
#include "common.h"
#include "cqueue.h"
#include "server.h"
#include "debug.h"
#include "env.h"
#include "handle.h"
#include "memory.h"
#include "window.h"
#include "quant.h"
#include "types.h"

cqueue_t** cqueue_table;
int num_cqueues;
int num_centrys;

MPI_Datatype default_mpi_types[NUM_MPI_TYPES] = {MPI_FLOAT, MPI_CHAR, MPI_UNSIGNED_CHAR, MPI_SIGNED_CHAR, \
						 MPI_SHORT, MPI_UNSIGNED_SHORT, MPI_INT, MPI_UNSIGNED, \
						 MPI_LONG, MPI_UNSIGNED_LONG, MPI_DOUBLE, MPI_LONG_DOUBLE, MPI_DATATYPE_NULL};
MPI_Op default_mpi_ops[NUM_MPI_OPS] = {MPI_SUM, MPI_MAX, MPI_MIN, MPI_PROD, MPI_LAND, MPI_BAND, MPI_LOR, \
				       MPI_BOR, MPI_LXOR, MPI_BXOR, MPI_QUANT_OP, MPI_OP_NULL};

static inline centry_t* cqueue_remove_head(cqueue_t*);

#ifdef ENABLE_FILEIO
static inline fentry_t* cqueue_remove_fhead(cqueue_t*);
#endif

#ifdef ENABLE_CQUEUE_ATOMIC
centry_t* cqueue_insert_tail(cqueue_t* mycqueue)
{
    centry_t* mycentry = NULL;
    unsigned int next_centry = __sync_add_and_fetch(&mycqueue->tail, 1) % CQUEUE_MAX_CENTRY;
    unsigned int temp_centry;

#ifndef ENABLE_ASYNC_PROGRESS
    while (mycqueue->centry_table[next_centry].cmd != CMD_EMPTY) {
    temp_centry = __sync_add_and_fetch(&mycqueue->tail, 1);
    next_centry = temp_centry % CQUEUE_MAX_CENTRY;
    }
#endif
    mycentry = &mycqueue->centry_table[next_centry];
    DEBUG_ASSERT(mycentry != NULL);

    return mycentry;
}

centry_t* cqueue_remove_head(cqueue_t* mycqueue)
{
    centry_t* mycentry = NULL;
    unsigned int temp_centry;

    /* Find a free slot */
#ifdef ENABLE_ASYNC_PROGRESS
    if (mycqueue->head != mycqueue->tail)
    {
        temp_centry = __sync_add_and_fetch(&mycqueue->head, 1);
        mycentry = &mycqueue->centry_table[temp_centry % CQUEUE_MAX_CENTRY];
        DEBUG_ASSERT(mycentry != NULL);
    }
#else
    int next_centry;
    while (mycqueue->head != mycqueue->tail)
    {
        temp_centry = __sync_add_and_fetch(&mycqueue->head, 1);
        next_centry = temp_centry % CQUEUE_MAX_CENTRY;
        if (mycqueue->centry_table[next_centry].cmd != CMD_EMPTY && mycqueue->centry_table[next_centry].cmd != CMD_ISSUED)
        {
            mycentry = &mycqueue->centry_table[next_centry];
            DEBUG_ASSERT(mycentry != NULL);
            break;
        }
    }
#endif
    return mycentry;
}

void cqueue_update_tail(cqueue_t* mycqueue)
{
    unsigned int tmp = __sync_add_and_fetch(&mycqueue->tail, 1);
}

#else /* No atomic updates */

#ifdef ENABLE_ASYNC_PROGRESS

#define cqueue_insert_tail1(mycqueue) &mycqueue->centry_table[(mycqueue->tail + 1) % CQUEUE_MAX_CENTRY]
#define cqueue_insert_tail(mycqueue) &mycqueue->centry_table[mycqueue->tail % CQUEUE_MAX_CENTRY]

static inline centry_t* cqueue_remove_head(cqueue_t* mycqueue)
{
    if (mycqueue->head != mycqueue->tail)
    {
        centry_t* mycentry = &mycqueue->centry_table[mycqueue->head % CQUEUE_MAX_CENTRY];
        mycqueue->head++;
        return mycentry;
    }
    return NULL;
}

#define cqueue_update_tail(mycqueue) mycqueue->tail++

#ifdef ENABLE_FILEIO

#define cqueue_insert_ftail(mycqueue) &mycqueue->fentry_table[(mycqueue->ftail + 1) % CQUEUE_MAX_FENTRY]

inline fentry_t* cqueue_remove_fhead(cqueue_t* mycqueue)
{
    if (mycqueue->fhead != mycqueue->ftail)
    {
        mycqueue->fhead++;
        return &mycqueue->fentry_table[mycqueue->fhead % CQUEUE_MAX_FENTRY];
    }
    return NULL;
}

#define cqueue_update_ftail(mycqueue) mycqueue->ftail++

inline fentry_t* cqueue_get_fentry(int fentryid)
{
    return &cqueue_table[fentryid/CQUEUE_MAX_FENTRY]->fentry_table[fentryid%CQUEUE_MAX_FENTRY];
}

#endif /* ENABLE_FILEIO */

#else /* !ENABLE_ASYNC_PROGRESS */

inline centry_t* cqueue_insert_tail(cqueue_t* mycqueue)
{
    centry_t* mycentry = NULL;
    int next_centry = (mycqueue->tail + 1) % CQUEUE_MAX_CENTRY;

#ifndef ENABLE_ASYNC_PROGRESS
    while (mycqueue->centry_table[next_centry].cmd != CMD_EMPTY)
    {
        mycqueue->tail++;
        next_centry = (mycqueue->tail + 1) % CQUEUE_MAX_CENTRY;
    }
#endif
    mycentry = &mycqueue->centry_table[next_centry];
    DEBUG_ASSERT(mycentry != NULL);

    return mycentry;
}

centry_t* cqueue_remove_head(cqueue_t* mycqueue)
{
    centry_t* mycentry = NULL;

#ifdef ENABLE_ASYNC_PROGRESS
    if (mycqueue->head != mycqueue->tail)
    {
        mycqueue->head++;
        mycentry = &mycqueue->centry_table[mycqueue->head % CQUEUE_MAX_CENTRY];
        DEBUG_ASSERT(mycentry != NULL);
    }
#else
    int next_centry;
    /* Find a free slot */
    while (mycqueue->head != mycqueue->tail)
    {
        mycqueue->head++;
        next_centry = mycqueue->head % CQUEUE_MAX_CENTRY;
        if (mycqueue->centry_table[next_centry].cmd != CMD_EMPTY && mycqueue->centry_table[next_centry].cmd != CMD_ISSUED)
        {
            mycentry = &mycqueue->centry_table[next_centry];
            DEBUG_ASSERT(mycentry != NULL);
            break;
        }
    }
#endif
    return mycentry;
}

inline void cqueue_update_tail(cqueue_t* mycqueue)
{
    mycqueue->tail++;
}

#endif /* ENABLE_ASYNC_PROGRESS */
#endif /* ENABLE_CQUEUE_ATOMIC */

#ifdef ENABLE_CLIENT_ONLY

void cqueue_init(int num_cq)
{
    /* Set number of command queues */
    num_cqueues = num_cq;

    /* Create data structure to track all command queues */
    MALLOC_ALIGN(cqueue_table, sizeof(cqueue_t*)*num_cqueues, CACHELINE_SIZE);

    /* Allocate in shared memory and align to page boundary */
    for (int cqueueid = 0; cqueueid < num_cqueues; cqueueid++)
    {
        cqueue_t* mycqueue = (cqueue_t*)memory_memalign(PAGE_SIZE, sizeof(cqueue_t));
        ASSERT(mycqueue);
        ASSERT_FMT((intptr_t) mycqueue % PAGE_SIZE == 0,
                   "mycqueue %% PAGE_SIZE == 0");
        mycqueue->head = 0;
        ASSERT_FMT((intptr_t) &mycqueue->head % CACHELINE_SIZE == 0,
                  "&mycqueue->head %% CACHELINE_SIZE == 0");
        mycqueue->tail = 0;
        ASSERT_FMT((intptr_t) &mycqueue->tail % CACHELINE_SIZE == 0,
                  "&mycqueue->tail %% CACHELINE_SIZE == 0");
        mycqueue->process = CQUEUE_EXECUTE;
        ASSERT_FMT((intptr_t) &mycqueue->process % CACHELINE_SIZE == 0,
                  "&mycqueue->process %% CACHELINE_SIZE == 0");
        for (int j = 0; j < CQUEUE_MAX_CENTRY; j++)
        {
            ASSERT_FMT((intptr_t) &mycqueue->centry_table[j] % CACHELINE_SIZE == 0,
                      "&mycqueue->centry_table[%d] %% CACHELINE_SIZE == 0", j);
            mycqueue->centry_table[j].centryid = (long) cqueueid*CQUEUE_MAX_CENTRY+j;
            mycqueue->centry_table[j].cmd = CMD_EMPTY;
            mycqueue->centry_table[j].datatype = MPI_DATATYPE_NULL;
            mycqueue->centry_table[j].datatype2 = MPI_DATATYPE_NULL;
            mycqueue->centry_table[j].op = MPI_OP_NULL;
            mycqueue->centry_table[j].is_prio_request = 0;
            num_centrys++;
        }
#ifdef ENABLE_FILEIO
        mycqueue->fhead = -1;
        ASSERT_FMT((intptr_t) &mycqueue->fhead % CACHELINE_SIZE == 0,
                   "&mycqueue->fhead %% CACHELINE_SIZE == 0");
        mycqueue->ftail = -1;
        ASSERT_FMT((intptr_t) &mycqueue->ftail % CACHELINE_SIZE == 0,
                  "&mycqueue->ftail %% CACHELINE_SIZE == 0");
        mycqueue->fprocess = CQUEUE_EXECUTE;
        ASSERT_FMT((intptr_t) &mycqueue->fprocess % CACHELINE_SIZE == 0,
                  "&mycqueue->fprocess %% CACHELINE_SIZE == 0");
        for (int j = 0; j < CQUEUE_MAX_FENTRY; j++)
        {
            ASSERT_FMT((intptr_t) &mycqueue->fentry_table[j] % CACHELINE_SIZE == 0,
                      "&mycqueue->fentry_table[%d] %% CACHELINE_SIZE == 0", j);
            mycqueue->fentry_table[j].fentryid = cqueueid*CQUEUE_MAX_FENTRY+j;
            mycqueue->fentry_table[j].cmd = CMD_EMPTY;
        }
#endif /* ENABLE_FILEIO */
        cqueue_table[cqueueid] = mycqueue;
        DEBUG_PRINT("client cqueue %p %d %ld %ld\n", mycqueue, cqueueid, sizeof(cqueue_t), sizeof(centry_t));
    }
}

cqueue_t* cqueue_attach(int cqueueid)
{
    return cqueue_table[cqueueid];
}

inline centry_t* cqueue_get_centry(long centryid)
{
    centry_t* mycentry = NULL;

    if (centryid >= 0 && centryid < num_centrys)
        mycentry = &cqueue_table[centryid/CQUEUE_MAX_CENTRY]->centry_table[centryid%CQUEUE_MAX_CENTRY];

    return mycentry;
}

#define CENTRY_GET_CQUEUE(centryid) cqueue_table[centryid / CQUEUE_MAX_CENTRY]

inline void cqueue_execute()
{
    for (int i = 0; i < num_cqueues; i++)
        cqueue_table[i]->process = CQUEUE_EXECUTE;
}

inline void cqueue_suspend()
{
    for (int i = 0; i < num_cqueues; i++)
        cqueue_table[i]->process = CQUEUE_SUSPEND;
}

inline int cqueue_comm_set_info(cqueue_t* mycqueue, MPI_Comm comm, const char* key, const char* value)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_COMM_SET_INFO;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->buffer = (void*)memory_calloc(1024, sizeof(char));
    mycentry->buffer2 = (void*)memory_calloc(1024, sizeof(char));
    memcpy((void*)mycentry->buffer, key, strlen(key));
    memcpy((void*)mycentry->buffer2, value, strlen(value));
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    memory_free((void*)mycentry->buffer);
    memory_free((void*)mycentry->buffer2);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

inline int cqueue_send(cqueue_t* mycqueue, const void* buffer, int count, MPI_Datatype datatype,
                       int tgt_rank, int tag, MPI_Comm comm)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_SEND;
    mycentry->buffer = (volatile void*)buffer;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->tgt_rank = tgt_rank;
    mycentry->tag = tag;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

inline int cqueue_recv(cqueue_t* mycqueue, void* buffer, int count, MPI_Datatype datatype,
                       int src_rank, int tag, MPI_Comm comm, MPI_Status* status)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_RECV;
    mycentry->buffer = (volatile void*)buffer;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->src_rank = src_rank;
    mycentry->tag = tag;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Update the status field */
    if (status != MPI_STATUS_IGNORE)
        memcpy((void*)status, (void*)&mycentry->status, sizeof(MPI_Status));

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

inline int cqueue_isend(cqueue_t* mycqueue, const void* buffer, int count, MPI_Datatype datatype,
                        int tgt_rank, int tag, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_ISEND;
    mycentry->buffer = (volatile void*)buffer;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->tgt_rank = tgt_rank;
    mycentry->tag = tag;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

inline int cqueue_irecv(cqueue_t* mycqueue, void* buffer, int count, MPI_Datatype datatype,
                        int src_rank, int tag, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IRECV;
    mycentry->buffer = (volatile void*)buffer;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->src_rank = src_rank;
    mycentry->tag = tag;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

#ifdef ENABLE_ASYNC_PROGRESS

inline int cqueue_wait(centry_t* mycentry, request_t* myrequest, MPI_Status* status)
{
    int ret;

    /* Release request object */
    REQUEST_SET(myrequest, MPI_REQUEST_NULL);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Update the status field */
    if (status != MPI_STATUS_IGNORE)
        memcpy((void*)status, (void*)&mycentry->status, sizeof(MPI_Status));

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

inline int cqueue_test(centry_t* mycentry, request_t* myrequest, int* flag, MPI_Status* status)
{
    int ret = MPI_SUCCESS;

    *flag = 0;

    /* Check if operation has completed */
    if (mycentry->ret != CMD_INFLIGHT) {
        /* Release request object */
        REQUEST_SET(myrequest, MPI_REQUEST_NULL);

        /* Set flag to mark completion */
        *flag = 1;

        /* Update the status field */
        if (status != MPI_STATUS_IGNORE)
            memcpy((void*)status, (void*)&mycentry->status, sizeof(MPI_Status));

        /* Set the entry as free */
        ret = mycentry->ret;
        mycentry->cmd = CMD_EMPTY;
    }

    return ret;
}

#else /* !ENABLE_ASYNC_PROGRESS */

inline int cqueue_wait(request_t* myrequest, centry_t* mycentry, MPI_Status* status)
{
    int ret;

    /* Release request object */
    REQUEST_SET(myrequest, MPI_REQUEST_NULL);

    cqueue_t* mycqueue = CENTRY_GET_CQUEUE(mycentry->centryid);

    /* Acquire a free entry from command queue */
    centry_t* newcentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    newcentry->buffer = (intptr_t) (volatile void*)&mycentry->request;
    newcentry->cmd = CMD_WAIT;
    newcentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (newcentry->ret == CMD_INFLIGHT);

    /* Update the status field */
    if (status != MPI_STATUS_IGNORE)
        memcpy((void*)status, (void*)&newcentry->status, sizeof(MPI_Status));

    /* Set the entry as free */
    ret = newcentry->ret;
    mycentry->cmd = CMD_EMPTY;
    newcentry->cmd = CMD_EMPTY;

    return ret;
}

inline int cqueue_test(request_t* myrequest, centry_t* mycentry, int* flag, MPI_Status* status)
{
    int ret;

    cqueue_t* mycqueue = CENTRY_GET_CQUEUE(mycentry->centryid);

    /* Acquire a free entry from command queue */
    centry_t* newcentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    newcentry->buffer = (volatile void*)&mycentry->request;
    newcentry->cmd = CMD_TEST;
    newcentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    if (newcentry->ret == CMD_INFLIGHT)
    {
        *flag = 0;
        return MPI_SUCCESS;
    }
    else
    {
        /* Release request object */
        REQUEST_SET(myrequest, MPI_REQUEST_NULL);

        /* Set flag to mark completion */
        *flag = 1;

        /* Update the status field */
        if (status != MPI_STATUS_IGNORE)
            memcpy((void*)status, (void*)&newcentry->status, sizeof(MPI_Status));

        /* Set the entry as free */
        ret = newcentry->ret;
        mycentry->cmd = CMD_EMPTY;
        newcentry->cmd = CMD_EMPTY;

        return ret;
    }
}

#endif /* ENABLE_ASYNC_PROGRESS */

int cqueue_barrier(cqueue_t* mycqueue, MPI_Comm comm)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_BARRIER;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_iallreduce(cqueue_t* mycqueue, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                      MPI_Op op, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IALLREDUCE;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
    }

    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->op = op;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_ialltoall(cqueue_t* mycqueue, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IALLTOALL;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
    }

    mycentry->count = sendcount;
    mycentry->datatype = sendtype;
    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count2 = recvcount;
    mycentry->datatype2 = recvtype;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_iallgather(cqueue_t* mycqueue, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IALLGATHER;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
        mycentry->count = sendcount;
        mycentry->datatype = sendtype;
    }

    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count2 = recvcount;
    mycentry->datatype2 = recvtype;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_igather(cqueue_t* mycqueue, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                   void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IGATHER;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
        mycentry->count = sendcount;
        mycentry->datatype = sendtype;
    }

    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count2 = recvcount;
    mycentry->datatype2 = recvtype;
    mycentry->root = root;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_ireduce_scatter_block(cqueue_t* mycqueue, const void* sendbuf, void* recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IREDUCE_SCATTER_BLOCK;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
    }

    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->op= op;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_ibcast(cqueue_t* mycqueue, void* recvbuf, int count,
                  MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IBCAST;
    mycentry->buffer = (volatile void*)recvbuf;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->root = root;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid / CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

int cqueue_ireduce(cqueue_t* mycqueue, const void* sendbuf, void* recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request* myrequest)
{
    centry_t* mycentry;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_IREDUCE;
    if (sendbuf == MPI_IN_PLACE)
        mycentry->inplace = 1;
    else
    {
        mycentry->inplace = 0;
        mycentry->buffer = (volatile void*)sendbuf;
    }
    mycentry->buffer2 = (volatile void*)recvbuf;
    mycentry->count = count;
    mycentry->datatype = datatype;
    mycentry->op= op;
    mycentry->root = root;
    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, mycentry->centryid);

    return MPI_SUCCESS;
}

#ifdef ENABLE_MPIRMA_ENDPOINTS

int cqueue_win_allocate(cqueue_t* mycqueue, MPI_Aint win_size, int disp_unit,
                        MPI_Info info, MPI_Comm comm, void* base_ptr, MPI_Win* win_ptr)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINALLOCATE;
    mycentry->buffer = (void*)memory_malloc(win_size);
    ASSERT(mycentry->buffer != NULL);
    mycentry->win_size = win_size;
    mycentry->disp_unit = disp_unit;

    if (info == MPI_INFO_NULL)
        mycentry->info = MPI_INFO_NULL;
    else
    {
        MPI_Info_create(&mycentry->info);
        memcpy(&info, &mycentry->info, sizeof(MPI_Info));
    }

    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid / CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Return the starting address and window object in server */
    *(void**)base_ptr = (void*)mycentry->buffer;
    *win_ptr = mycentry->win;

    /* Free info object if created earlier */
    if (info != MPI_INFO_NULL)
        MPI_Info_free(&mycentry->info);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_create(cqueue_t* mycqueue, void* base, MPI_Aint win_size, int disp_unit,
                      MPI_Info info, MPI_Comm comm, MPI_Win* win_ptr)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    ASSERT(memory_is_shmem(base, NULL));
    mycentry->cmd = CMD_WINCREATE;
    mycentry->buffer = (void*)base;
    ASSERT(mycentry->buffer != NULL);
    mycentry->win_size = win_size;
    mycentry->disp_unit = disp_unit;

    if (info == MPI_INFO_NULL)
    mycentry->info = MPI_INFO_NULL;
    else
    {
        MPI_Info_create(&mycentry->info);
        memcpy(&info, &mycentry->info, sizeof(MPI_Info));
    }

    mycentry->comm = handle_get_server_comm(comm, mycentry->centryid/CQUEUE_MAX_CENTRY);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Return the window object in server */
    *win_ptr = mycentry->win;

    /* Free info object if created earlier */
    if (info != MPI_INFO_NULL)
        MPI_Info_free(&mycentry->info);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_free(cqueue_t* mycqueue, MPI_Win* win_ptr)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINFREE;
    mycentry->win = window_get_server_win(*win_ptr);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set window object as free */
    *win_ptr = MPI_WIN_NULL;

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_put(cqueue_t* mycqueue, const void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
               int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_PUT;
    mycentry->buffer = (volatile void*)origin_addr;
    mycentry->count = origin_count;
    mycentry->datatype = origin_datatype;
    mycentry->tgt_rank = target_rank;
    mycentry->target_disp = target_disp;
    mycentry->target_count = target_count;
    mycentry->target_datatype = target_datatype;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_get(cqueue_t* mycqueue, void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
               int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_GET;
    mycentry->buffer = (volatile void*)origin_addr;
    mycentry->count = origin_count;
    mycentry->datatype = origin_datatype;
    mycentry->tgt_rank = target_rank;
    mycentry->target_disp = target_disp;
    mycentry->target_count = target_count;
    mycentry->target_datatype = target_datatype;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_lock(cqueue_t* mycqueue, int lock_type, int tgt_rank, int assert, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINLOCK;
    mycentry->lock_type = lock_type;
    mycentry->tgt_rank = tgt_rank;
    mycentry->assert = assert;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_unlock(cqueue_t* mycqueue, int tgt_rank, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINUNLOCK;
    mycentry->tgt_rank = tgt_rank;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_lock_all(cqueue_t* mycqueue, int assert, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINLOCKALL;
    mycentry->assert = assert;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_unlock_all(cqueue_t* mycqueue, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINUNLOCKALL;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_flush(cqueue_t* mycqueue, int tgt_rank, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINFLUSH;
    mycentry->tgt_rank = tgt_rank;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_flush_local(cqueue_t* mycqueue, int tgt_rank, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINFLUSHLOCAL;
    mycentry->tgt_rank = tgt_rank;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_flush_all(cqueue_t* mycqueue, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINFLUSHALL;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_win_fence(cqueue_t* mycqueue, int assert, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_WINFENCE;
    mycentry->assert = assert;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_compare_and_swap(cqueue_t* mycqueue, const void* origin_addr, const void* compare_addr,
                            void* result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_COMPARESWAP;
    mycentry->buffer = (volatile void*)origin_addr;
    mycentry->buffer2 = (volatile void*)compare_addr;
    mycentry->buffer3 = (volatile void*)result_addr;
    mycentry->datatype = datatype;
    mycentry->tgt_rank = target_rank;
    mycentry->target_disp = target_disp;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_fetch_and_op(cqueue_t* mycqueue, const void* origin_addr, void* result_addr, MPI_Datatype datatype,
                        int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)
{
    centry_t* mycentry;
    int ret;

    /* Acquire a free entry from command queue */
    mycentry = cqueue_insert_tail(mycqueue);

    /* Fill the entry with relevant details */
    mycentry->cmd = CMD_FETCHOP;
    mycentry->buffer = (volatile void*)origin_addr;
    mycentry->buffer2 = (volatile void*)result_addr;
    mycentry->datatype = datatype;
    mycentry->tgt_rank = target_rank;
    mycentry->target_disp = target_disp;
    mycentry->op = op;
    mycentry->win = window_get_server_win(win);
    mycentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_tail(mycqueue);

    /* Wait for operation to complete */
    while (mycentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = mycentry->ret;
    mycentry->cmd = CMD_EMPTY;

    return ret;
}

#endif /* ENABLE_MPIRMA_ENDPOINTS */

int cqueue_create_server_comm(MPI_Comm comm, int color, int key, MPI_Comm* worldcomm, MPI_Comm* peercomm)
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t *myrequest;

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->cmd = CMD_COMM_SPLIT;
        mycentry->comm = handle_get_server_worldcomm(comm, epid);
        mycentry->color = color;
        mycentry->key = key;
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT);

        /* Set the entry as free */
        worldcomm[epid] = mycentry->comm;
        mycentry->cmd = CMD_EMPTY;
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->cmd = CMD_COMM_CREATE_PEER;
        mycentry->comm = worldcomm[epid];
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT);

        /* Set the entry as free */
        peercomm[epid] = mycentry->comm;
        mycentry->cmd = CMD_EMPTY;
    }

    free(myrequest);

    return MPI_SUCCESS;
}

int cqueue_comm_free(MPI_Comm comm)
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t *myrequest;

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

	/* Fill the entry with relevant details */
	mycentry->cmd = CMD_COMM_FREE;
	mycentry->comm = handle_get_server_comm(comm, epid);
	mycentry->ret = CMD_INFLIGHT;

	/* Update the entry to indicate it is ready for processing */
	cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT);

        /* Set the entry as free */
        mycentry->cmd = CMD_EMPTY;
    }

    free(myrequest);

    return MPI_SUCCESS;
}

int cqueue_memory_register(void* baseaddr, size_t size_bytes, int* memid_ptr)
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t* myrequest;
    int    memid;

    memid = memory_register(baseaddr, size_bytes, uuid_str, 1 /* client */);
    if (memid == -1)
        ERROR("Client failed to register memory region at %p (%ld bytes)\n", baseaddr, size_bytes);

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->cmd = CMD_MEMORY_REGISTER;
        mycentry->buffer = baseaddr;
        mycentry->comm = handle_get_server_worldcomm(MPI_COMM_WORLD, epid);
        mycentry->memid = -1;
        mycentry->size_bytes = size_bytes;
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT && mycentry->ret != CMD_SUCCESS);

        /* Set the entry as free */
        if (mycentry->memid == -1) memid = -1;
        mycentry->cmd = CMD_EMPTY;
    }

    free(myrequest);

    if (memid == -1)
        ERROR("Unable to register memory region at %p (%ld bytes)\n", baseaddr, size_bytes);

    *memid_ptr = memid;
    return MPI_SUCCESS;
}

int cqueue_memory_release(int memid)
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t* myrequest;

    memory_release(memid);

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->cmd = CMD_MEMORY_RELEASE;
        mycentry->comm = handle_get_server_worldcomm(MPI_COMM_WORLD, epid);
        mycentry->memid = memid;
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT && mycentry->ret != CMD_SUCCESS);

        /* Set the entry as free */
        mycentry->cmd = CMD_EMPTY;
    }

    free(myrequest);
    return MPI_SUCCESS;
}

int cqueue_mpi_type_register()
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t* myrequest;
    MPI_Datatype* mpitype;

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    mpitype = (MPI_Datatype*) memory_malloc(NUM_MPI_TYPES * sizeof(MPI_Datatype));
    memcpy(mpitype, default_mpi_types, NUM_MPI_TYPES * sizeof(MPI_Datatype));

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->buffer = mpitype;
        mycentry->buffer2 = (void*)MPI_DATATYPE_NULL;
        mycentry->cmd = CMD_CONVERT_MPI_TYPES;
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT);

        /* Set the entry as free */
        mycentry->cmd = CMD_EMPTY;
    }

    memory_free(mpitype);
    free(myrequest);

    return MPI_SUCCESS;
}

int cqueue_mpi_op_register()
{
    cqueue_t* mycqueue;
    centry_t* mycentry;
    request_t* myrequest;
    MPI_Op* mpiop;

    myrequest = (request_t*)malloc(max_ep * sizeof(request_t));
    ASSERT(myrequest);

    mpiop = (MPI_Op*) memory_malloc(NUM_MPI_OPS * sizeof(MPI_Op));
    memcpy(mpiop, default_mpi_ops, NUM_MPI_OPS * sizeof(MPI_Op));

    for (int epid = 0; epid < max_ep; epid++)
    {
        /* Get the command queue for this endpoint */
        mycqueue = cqueue_table[epid];

        /* Acquire a free entry from command queue */
        mycentry = cqueue_insert_tail(mycqueue);

        /* Fill the entry with relevant details */
        mycentry->buffer = mpiop;
        mycentry->buffer2 = (void*)MPI_OP_NULL;
        mycentry->cmd = CMD_CONVERT_MPI_OPS;
        mycentry->ret = CMD_INFLIGHT;

        /* Update the entry to indicate it is ready for processing */
        cqueue_update_tail(mycqueue);

        /* Register request object */
        REQUEST_SET(&myrequest[epid], mycentry->centryid);
    }

    for (int epid = 0; epid < max_ep; epid++)
    {
        mycentry = cqueue_get_centry((long)myrequest[epid]);

        /* Release request object */
        REQUEST_SET(&myrequest[epid], MPI_REQUEST_NULL);

        /* Wait for operation to complete */
        while(mycentry->ret == CMD_INFLIGHT);

        /* Set the entry as free */
        mycentry->cmd = CMD_EMPTY;
    }

    memory_free(mpiop);
    free(myrequest);

    return MPI_SUCCESS;
}

#ifdef ENABLE_FILEIO

int cqueue_fopen(cqueue_t* mycqueue, const char* filename, const char* mode, FILE** stream)
{
    fentry_t* myfentry;
    int ret;

    /* Acquire a free entry from command queue */
    myfentry = cqueue_insert_ftail(mycqueue);

    /* Fill the entry with relevant details */
    myfentry->cmd = CMD_FOPEN;
    strcpy((char*)myfentry->filename, filename);
    strcpy((char*)myfentry->mode, mode);
    myfentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_ftail(mycqueue);

    /* Wait for operation to complete */
    while (myfentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    *stream = (FILE*)myfentry->stream;
    ret = myfentry->ret;
    myfentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_fread_nb(cqueue_t* mycqueue, void* buffer, size_t size, size_t count, FILE* stream, MPI_Request* myrequest)
{
    fentry_t* myfentry;
    int ret;

    /* Acquire a free entry from command queue */
    myfentry = cqueue_insert_ftail(mycqueue);

    /* Fill the entry with relevant details */
    myfentry->cmd = CMD_FREAD_NB;
    myfentry->buffer = (volatile void*)buffer;
    myfentry->size = size;
    myfentry->count = count;
    myfentry->stream = (volatile FILE*)stream;
    myfentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_ftail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, myfentry->fentryid);

    return MPI_SUCCESS;
}

int cqueue_forc_nb(cqueue_t* mycqueue, const char* filename, const char* mode, void* buffer,
                   size_t size, size_t count, MPI_Request* myrequest)
{
    fentry_t* myfentry;
    int ret;

    /* Acquire a free entry from command queue */
    myfentry = cqueue_insert_ftail(mycqueue);

    /* Fill the entry with relevant details */
    myfentry->cmd = CMD_FORC_NB;
    strcpy((char*)myfentry->filename, filename);
    strcpy((char*)myfentry->mode, mode);
    myfentry->buffer = (volatile void*)buffer;
    myfentry->size = size;
    myfentry->count = count;
    myfentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_ftail(mycqueue);

    /* Register request object */
    REQUEST_SET(myrequest, myfentry->fentryid);

    return MPI_SUCCESS;
}

inline int cqueue_fwait(fentry_t* myfentry, request_t* myrequest, size_t* readsize)
{
    int ret;

    /* Release request object */
    REQUEST_SET(myrequest, MPI_REQUEST_NULL);

    /* Wait for operation to complete */
    while (myfentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    *readsize = myfentry->size;
    ret = myfentry->ret;
    myfentry->cmd = CMD_EMPTY;

    return ret;
}

int cqueue_fclose(cqueue_t* mycqueue, FILE* stream)
{
    fentry_t* myfentry;
    int ret;

    /* Acquire a free entry from command queue */
    myfentry = cqueue_insert_ftail(mycqueue);

    /* Fill the entry with relevant details */
    myfentry->cmd = CMD_FCLOSE;
    myfentry->stream = (FILE*)stream;
    myfentry->ret = CMD_INFLIGHT;

    /* Update the entry to indicate it is ready for processing */
    cqueue_update_ftail(mycqueue);

    /* Wait for operation to complete */
    while (myfentry->ret == CMD_INFLIGHT);

    /* Set the entry as free */
    ret = myfentry->ret;
    myfentry->cmd = CMD_EMPTY;

    return ret;
}

#endif /* ENABLE_FILEIO */

void cqueue_finalize()
{
    centry_t** centry_table;

    centry_table = (centry_t**)malloc(sizeof(centry_t*) * num_cqueues);
    ASSERT(centry_table);

    for (int i = 0; i < num_cqueues; i++)
    {
        cqueue_t* mycqueue = cqueue_table[i];
        cqueue_table[i]->process = CQUEUE_EXECUTE;
        centry_table[i] = cqueue_insert_tail(mycqueue);
        centry_table[i]->cmd = CMD_FINALIZE;
        centry_table[i]->ret = CMD_INFLIGHT;
        cqueue_update_tail(mycqueue);
    }

    for (int i = 0; i < num_cqueues; i++)
    {
        cqueue_t* mycqueue = cqueue_table[i];
        while(centry_table[i]->ret == CMD_INFLIGHT);
        memory_free(mycqueue);
    }

    free(centry_table);
    FREE_ALIGN(cqueue_table);
}

#else /* ! ENABLE_CLIENT_ONLY */

#include "server.h"

#endif /* ENABLE_CLIENT_ONLY */

MPI_Datatype convert_mpi_type(MPI_Datatype datatype_table[NUM_MPI_TYPES], MPI_Datatype datatype)
{
    for (int i = 0; i < NUM_MPI_TYPES; i++)
	if (datatype_table[i] == datatype)
	    return default_mpi_types[i];
    ERROR("MPI Datatype %ld not registered with server\n", (long)datatype);
}

MPI_Op convert_mpi_op(MPI_Op op_table[NUM_MPI_OPS], MPI_Op op)
{
    for (int i = 0; i < NUM_MPI_OPS; i++)
	if (op_table[i] == op)
	    return default_mpi_ops[i];
    ERROR("MPI Op %ld not registered with server\n", (long)op);
}

void cqueue_process(cqueue_t* mycqueue, MPI_Comm taskcomm)
{
    centry_t* mycentry = NULL;
    int test_flag, done_flag = 0;
    int cmd;
    MPI_Comm newcomm;
    int serverid = -1, epid = -1;
    int rank, numranks;
    int newserverid, newserversize;
    quant_lib_status q_lib_status = quant_lib_unloaded;
    MPI_Op reduce_op;
    int    reduce_count;
    MPI_Info info;

    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &numranks);

    if (msg_priority) allreduce_pr_initialize();

#if 0
    int sb = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    if (rank == 1) {
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == sb)
        sleep(5);
    }
#endif

    int num_pending = 0;
    int pid, max_pending;
    centry_t* pending_list[CQUEUE_MAX_CENTRY];
    max_pending = CQUEUE_MAX_CENTRY;
    for (pid = 0; pid < max_pending; pid++)
        pending_list[pid] = NULL;

    /* Post dummy Irecv to ensure communication progress even if there is no pending commands */
    MPI_Request dummy_request;
    MPI_Comm dummy_comm;
    PMPI_Comm_dup(MPI_COMM_WORLD, &dummy_comm);
    PMPI_Irecv(NULL, 0, MPI_CHAR, 0, 0, dummy_comm, &dummy_request);

    if (dynamic_server == DYNAMIC_SERVER_ASYNCTHREAD)
        mycqueue->process = CQUEUE_SUSPEND;

    /* Conversion of client to server datatypes/ops */
    int convert_types = 0;
    int convert_ops = 0;
    MPI_Datatype mpi_type_table[NUM_MPI_TYPES];
    MPI_Op mpi_op_table[NUM_MPI_OPS];
    MPI_Datatype null_datatype = MPI_DATATYPE_NULL;
    MPI_Op null_op = MPI_OP_NULL;

    int master_proc_count = numranks / max_ep;
    char power_of_two = ((master_proc_count & (master_proc_count - 1)) == 0);

    while (!done_flag)
    {
        if (mycqueue->process == CQUEUE_EXECUTE)
        {
            /* Check for new commands from client */
            mycentry = NULL;
            mycentry = cqueue_remove_head(mycqueue);

            if (mycentry != NULL && mycentry->cmd != CMD_ISSUED && mycentry->cmd != CMD_EMPTY)
            {
                int nbcmd = 0;

                /* New command inserted into command queue */
                cmd = mycentry->cmd;
                mycentry->cmd = CMD_ISSUED;
		if (convert_types == 1)
		{
		    mycentry->datatype = convert_mpi_type(mpi_type_table, mycentry->datatype);
		    mycentry->datatype2 = convert_mpi_type(mpi_type_table, mycentry->datatype2);
		}
		if (convert_ops == 1)
		    mycentry->op = convert_mpi_op(mpi_op_table, mycentry->op);
                switch (cmd)
                {
                    case CMD_ISEND:
                        PMPI_Isend(memory_translate_clientaddr((void*)mycentry->buffer), mycentry->count,
				   mycentry->datatype, mycentry->tgt_rank,
				   mycentry->tag, mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IRECV:
                        PMPI_Irecv(memory_translate_clientaddr((void*)mycentry->buffer), mycentry->count,
				   mycentry->datatype, mycentry->src_rank,
				   mycentry->tag, mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_TEST:
                        test_flag = 0;
                        PMPI_Test((MPI_Request*)memory_translate_clientaddr((void*)mycentry->buffer),
				  &test_flag, (MPI_Status*)&mycentry->status);
                        if (test_flag == 1 && mycentry->op == MPI_QUANT_OP)
                        {
                            quant_dequantize(memory_translate_clientaddr((void*)mycentry->buffer2), mycentry->count);
                        }
                        if (test_flag == 1) mycentry->ret = MPI_SUCCESS;
                        break;
                    case CMD_WAIT:
                        mycentry->ret = PMPI_Wait((MPI_Request*)memory_translate_clientaddr((void*)mycentry->buffer),
                                                  (MPI_Status*)&mycentry->status);
                        if (mycentry->op == MPI_QUANT_OP)
                        {
                            quant_dequantize(memory_translate_clientaddr((void*)mycentry->buffer2), mycentry->count);
                        }
                        break;
                    case CMD_SEND:
                        mycentry->ret = PMPI_Send(memory_translate_clientaddr((void*)mycentry->buffer),
						  mycentry->count, mycentry->datatype,
						  mycentry->tgt_rank, mycentry->tag, mycentry->comm);
                        break;
                    case CMD_RECV:
                        mycentry->ret = PMPI_Recv(memory_translate_clientaddr((void*)mycentry->buffer),
						  mycentry->count, mycentry->datatype,
						  mycentry->src_rank, mycentry->tag,
						  mycentry->comm, (MPI_Status*)&mycentry->status);
                        break;
                    case CMD_BARRIER:
                        mycentry->ret = PMPI_Barrier(mycentry->comm);
                        break;
                    case CMD_IALLREDUCE:
                        reduce_op = mycentry->op;
                        reduce_count = mycentry->count;
                        if (mycentry->op == MPI_QUANT_OP)
                        {
                            void* quant_buf;
                            if (q_lib_status == quant_lib_unloaded)
                            {
                                q_lib_status = quant_lib_loaded;
                                quant_params_t* ptr_qparam = (quant_params_t*)memory_get_quant_params(0);
                                quant_init(ptr_qparam);
                            }
                            reduce_op = quant_get_op();
                            mycentry->datatype = quant_get_data_type();
                            if (mycentry->inplace == 1)
                                quant_buf = memory_translate_clientaddr((void*)mycentry->buffer2);
                            else
                                quant_buf = memory_translate_clientaddr((void*)mycentry->buffer);
                            quant_quantize(quant_buf ,mycentry->count);
                            reduce_count = quant_get_reduce_count(mycentry->count);
                        }

                        int type_size;
                        MPI_Type_size(mycentry->datatype, &type_size);

                        if (power_of_two && msg_priority &&
                            (reduce_count * type_size) > msg_priority_threshold)
                        {
                            if (mycentry->inplace == 1)
                                allreduce_pr_start(MPI_IN_PLACE, memory_translate_clientaddr((void*)mycentry->buffer2),
                                                   reduce_count, mycentry->datatype, reduce_op,
                                                   mycentry->comm, &mycentry->request);
                            else
                                allreduce_pr_start(memory_translate_clientaddr((void*)mycentry->buffer),
                                                   memory_translate_clientaddr((void*)mycentry->buffer2),
                                                   reduce_count, mycentry->datatype, reduce_op,
                                                   mycentry->comm, &mycentry->request);
                            mycentry->is_prio_request = 1;
                        }
                        else
                        {
                          if (mycentry->inplace == 1)
                              PMPI_Iallreduce(MPI_IN_PLACE, memory_translate_clientaddr((void*)mycentry->buffer2),
                                              reduce_count/1, mycentry->datatype, reduce_op,
                                              mycentry->comm, &mycentry->request);
                          else
                              PMPI_Iallreduce(memory_translate_clientaddr((void*)mycentry->buffer),
                                              memory_translate_clientaddr((void*)mycentry->buffer2),
                                              reduce_count/1, mycentry->datatype, reduce_op,
                                              mycentry->comm, &mycentry->request);
                        }
                        nbcmd = 1;
                        break;
                    case CMD_IALLTOALL:
                        if (mycentry->inplace == 1)
                            PMPI_Ialltoall(MPI_IN_PLACE, mycentry->count, mycentry->datatype,
					   memory_translate_clientaddr((void*)mycentry->buffer2),
					   mycentry->count2, mycentry->datatype2,
					   mycentry->comm, &mycentry->request);
                        else
                            PMPI_Ialltoall(memory_translate_clientaddr((void*)mycentry->buffer),
					   mycentry->count, mycentry->datatype,
					   memory_translate_clientaddr((void*)mycentry->buffer2),
					   mycentry->count2, mycentry->datatype2,
					   mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IALLGATHER:
                        if (mycentry->inplace == 1)
                            PMPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
					    memory_translate_clientaddr((void*)mycentry->buffer2),
					    mycentry->count2, mycentry->datatype2,
					    mycentry->comm, &mycentry->request);
                        else
                            PMPI_Iallgather(memory_translate_clientaddr((void*)mycentry->buffer),
					    mycentry->count, mycentry->datatype,
					    memory_translate_clientaddr((void*)mycentry->buffer2),
					    mycentry->count2, mycentry->datatype2,
					    mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IGATHER:
                        if (mycentry->inplace == 1)
                            PMPI_Igather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
					 memory_translate_clientaddr((void*)mycentry->buffer2),
					 mycentry->count2, mycentry->datatype2, mycentry->root,
					 mycentry->comm, &mycentry->request);
                        else
                            PMPI_Igather(memory_translate_clientaddr((void*)mycentry->buffer),
					 mycentry->count, mycentry->datatype,
					 memory_translate_clientaddr((void*)mycentry->buffer2),
					 mycentry->count2, mycentry->datatype2, mycentry->root,
					 mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IREDUCE_SCATTER_BLOCK:
                        if (mycentry->inplace == 1)
                            PMPI_Ireduce_scatter_block(MPI_IN_PLACE, memory_translate_clientaddr((void*)mycentry->buffer2),
						       mycentry->count, mycentry->datatype, mycentry->op,
						       mycentry->comm, &mycentry->request);
                        else
                            PMPI_Ireduce_scatter_block(memory_translate_clientaddr((void*)mycentry->buffer),
						       memory_translate_clientaddr((void*)mycentry->buffer2),
						       mycentry->count, mycentry->datatype, mycentry->op,
						       mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IBCAST:
                        PMPI_Ibcast(memory_translate_clientaddr((void*)mycentry->buffer),
				    mycentry->count, mycentry->datatype, mycentry->root,
				    mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
                    case CMD_IREDUCE:
                        if (mycentry->inplace == 1)
			    PMPI_Ireduce(MPI_IN_PLACE, memory_translate_clientaddr((void*)mycentry->buffer2),
					 mycentry->count, mycentry->datatype, mycentry->op,
					 mycentry->root, mycentry->comm, &mycentry->request);
			else
			    PMPI_Ireduce(memory_translate_clientaddr((void*)mycentry->buffer),
					 memory_translate_clientaddr((void*)mycentry->buffer2),
					 mycentry->count, mycentry->datatype, mycentry->op,
					 mycentry->root, mycentry->comm, &mycentry->request);
                        nbcmd = 1;
                        break;
#ifdef ENABLE_MPIRMA_ENDPOINTS
                    case CMD_WINALLOCATE:
                    case CMD_WINCREATE:
                        mycentry->ret = PMPI_Win_create((void*)mycentry->buffer, mycentry->win_size,
                                                        mycentry->disp_unit, mycentry->info,
                                                        mycentry->comm, &(MPI_Win)mycentry->win);
                        break;
                    case CMD_PUT:
                        mycentry->ret = PMPI_Put((void*)mycentry->buffer, mycentry->count,
						 mycentry->datatype, mycentry->tgt_rank,
						 mycentry->target_disp, mycentry->target_count,
						 mycentry->target_datatype, mycentry->win);
                        break;
                    case CMD_GET:
                        mycentry->ret = PMPI_Get((void*)mycentry->buffer, mycentry->count,
						 mycentry->datatype, mycentry->tgt_rank,
						 mycentry->target_disp, mycentry->target_count,
						 mycentry->target_datatype, mycentry->win);
                        break;
                    case CMD_WINLOCK:
                        mycentry->ret = PMPI_Win_lock(mycentry->lock_type, mycentry->tgt_rank,
                                                      mycentry->assert, mycentry->win);
                        break;
                    case CMD_WINUNLOCK:
                        mycentry->ret = PMPI_Win_unlock(mycentry->tgt_rank, mycentry->win);
                        break;
                    case CMD_WINLOCKALL:
                        mycentry->ret = PMPI_Win_lock_all(mycentry->assert, mycentry->win);
                        break;
                    case CMD_WINUNLOCKALL:
                        mycentry->ret = PMPI_Win_unlock_all(mycentry->win);
                        break;
                    case CMD_WINFLUSH:
                        mycentry->ret = PMPI_Win_flush(mycentry->tgt_rank, mycentry->win);
                        break;
                    case CMD_WINFLUSHLOCAL:
                        mycentry->ret = PMPI_Win_flush_local(mycentry->tgt_rank, mycentry->win);
                        break;
                    case CMD_WINFLUSHALL:
                        mycentry->ret = PMPI_Win_flush_all(mycentry->win);
                        break;
                    case CMD_WINFENCE:
                        mycentry->ret = PMPI_Win_fence(mycentry->assert, mycentry->win);
                        break;
                    case CMD_WINFREE:
                        mycentry->ret = PMPI_Win_free(&(MPI_Win)mycentry->win);
                        break;
                    case CMD_COMPARESWAP:
                        mycentry->ret = PMPI_Compare_and_swap((void*)mycentry->buffer, (void*)mycentry->buffer2,
							      (void*)mycentry->buffer3, mycentry->datatype,
							      mycentry->tgt_rank, mycentry->target_disp, mycentry->win);
                        break;
                    case CMD_FETCHOP:
                        mycentry->ret = PMPI_Fetch_and_op((void*)mycentry->buffer, (void*)mycentry->buffer2,
							  mycentry->datatype, mycentry->tgt_rank,
							  mycentry->target_disp, mycentry->op, mycentry->win);
                        break;
#endif /* ENABLE_MPIRMA_ENDPOINTS */
                    case CMD_COMM_SPLIT:
                        if (mycentry->comm == EPLIB_COMM_NULL)
                            newcomm = MPI_COMM_WORLD;
                        else
                        {
                            PMPI_Comm_rank(mycentry->comm, &serverid);
                            epid = serverid % max_ep;
                            PMPI_Comm_split(mycentry->comm, mycentry->color, mycentry->key*max_ep+epid, &newcomm);
                        }
                        PMPI_Comm_rank(newcomm, &newserverid);
                        PMPI_Comm_size(newcomm, &newserversize);
                        mycentry->comm = newcomm;
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_COMM_CREATE_PEER:
                        PMPI_Comm_rank(mycentry->comm, &serverid);
                        epid = serverid % max_ep;
                        PMPI_Comm_split(mycentry->comm, epid, serverid, &newcomm);
                        PMPI_Comm_rank(newcomm, &newserverid);
                        PMPI_Comm_size(newcomm, &newserversize);
                        mycentry->comm = newcomm;
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_COMM_FREE:
                        PMPI_Comm_free((MPI_Comm*)&(mycentry->comm));
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_COMM_SET_INFO:
                        PMPI_Info_create(&info);
                        PMPI_Info_set(info, memory_translate_clientaddr((void*)mycentry->buffer),
                                            memory_translate_clientaddr((void*)mycentry->buffer2));
                        PMPI_Comm_set_info(mycentry->comm, info);
                        PMPI_Info_free(&info);
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_EMPTY:
                        break;
                    case CMD_ISSUED:
                        ERROR("Increase queue size\n");
                        break;
                    case CMD_MEMORY_REGISTER:
                        mycentry->memid = server_memory_register((void*)mycentry->buffer, mycentry->size_bytes, taskcomm);
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_MEMORY_RELEASE:
                        memory_release(mycentry->memid);
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_CONVERT_MPI_TYPES:
                        /* Copy client and server types to idx 0 and 1 respectively */
                        memcpy(mpi_type_table, memory_translate_clientaddr((void*)mycentry->buffer), NUM_MPI_TYPES * sizeof(MPI_Datatype));
                        for (int typeid = 0; typeid < NUM_MPI_TYPES; typeid++)
                            if (mpi_type_table[typeid] != default_mpi_types[typeid])
                            {
                                convert_types = 1;
                                null_datatype = (MPI_Datatype)((long)mycentry->buffer2);
                                break;
                            }
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_CONVERT_MPI_OPS:
                        memcpy(mpi_op_table, memory_translate_clientaddr((void*)mycentry->buffer), NUM_MPI_OPS * sizeof(MPI_Op));
                        for (int opid = 0; opid < NUM_MPI_OPS; opid++)
                            if (mpi_op_table[opid] != default_mpi_ops[opid])
                            {
                                convert_ops = 1;
                                null_op = (MPI_Op)((long)mycentry->buffer2);
                                break;
                            }
                        mycentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_FINALIZE:
                        PMPI_Cancel(&dummy_request);
                        PMPI_Comm_free(&dummy_comm);

                        PMPI_Barrier(MPI_COMM_WORLD);
                        mycentry->ret = MPI_SUCCESS;
                        done_flag = 1;
                        if (q_lib_status == quant_lib_loaded)
                        {
                            quant_finalize();
                        }
                        return;
                        break;
                    default:
                        printf("Server cannot process command %d\n", cmd);
                        break;
                } // switch

                if (nbcmd == 1)
                {
                    nbcmd = 0;
                    for (pid = 0; pid < max_pending; pid++)
                    {
                        if (pending_list[pid] == NULL)
                        {
                            pending_list[pid] = mycentry;
                            num_pending++;
                            break;
                        }
                    }
                }

                mycentry->datatype = null_datatype;
                mycentry->datatype2 = null_datatype;
                mycentry->op = null_op;

            } // if (mycentry != NULL ...
        } // if (mycqueue->process)

        if (num_pending > 0)
        {
            int cur_pending = num_pending;
            for (pid = 0; pid < max_pending; pid++)
            {
                if (pending_list[pid] != NULL)
                {
                    mycentry = pending_list[pid];
                    test_flag = 0;
                    if (mycentry->is_prio_request)
                        allreduce_pr_test(&mycentry->request, (int*)&test_flag);
                    else
                        PMPI_Test((MPI_Request*)&mycentry->request, &test_flag, (MPI_Status*)&mycentry->status);
                    if (test_flag)
                    {
                        mycentry->is_prio_request = 0;
                        if (mycentry->op == MPI_QUANT_OP)
                            quant_dequantize(memory_translate_clientaddr((void*)mycentry->buffer2), mycentry->count);

                        mycentry->ret = MPI_SUCCESS;
                        num_pending--;
                        pending_list[pid] = NULL;
                    }
                    cur_pending--;
                }
                if (num_pending == 0 || cur_pending == 0) break;
            }
        }
        else
        {
            /* ensure communication progress even if there is no pending commands */
            PMPI_Test(&dummy_request, &test_flag, MPI_STATUS_IGNORE);
        }

#ifdef ENABLE_FILEIO
        int fd;
        FILE* stream;
        fentry_t* myfentry = NULL;
        if (mycqueue->fprocess)
        {
            /* Check for new commands from client */
            myfentry = NULL;
            myfentry = cqueue_remove_fhead(mycqueue);

            if (myfentry)
            {
                /* New command inserted into command queue */
                cmd = myfentry->cmd;
                myfentry->cmd = CMD_ISSUED;
                switch (cmd)
                {
                    case CMD_FOPEN:
                        myfentry->stream = fopen((const char*)myfentry->filename, (const char*)myfentry->mode);
                        myfentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_FREAD_NB:
                        myfentry->size = fread(memory_translate_clientaddr((void*)myfentry->buffer),
                                               myfentry->size, myfentry->count, (FILE*)myfentry->stream);
                        myfentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_FORC_NB:
                        fd = open((const char*)myfentry->filename, O_DIRECT);
                        stream = fdopen(fd, (const char*)myfentry->mode);
                        myfentry->size = fread(memory_translate_clientaddr((void*)myfentry->buffer),
                                               myfentry->size, myfentry->count, stream);
                        fclose(stream);
                        close(fd);
                        myfentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_FCLOSE:
                        fclose((FILE*)myfentry->stream);
                        myfentry->ret = CMD_SUCCESS;
                        break;
                    case CMD_EMPTY:
                        break;
                    case CMD_ISSUED:
                        ERROR("Increase i-queue size\n");
                        break;
                    default:
                        printf("Server cannot process i-command %d\n", cmd);
                        break;
                }
            } // if (myfentry)
        } // if (mycqueue->fprocess)
#endif /* ENABLE_FILEIO */

    } // while (!done_flag)
}
