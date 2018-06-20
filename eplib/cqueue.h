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
#ifndef _CQUEUE_H_
#define _CQUEUE_H_

#include "common.h"
#include "request.h"

#define CQUEUE_MAX_CENTRY         1000
#define CQUEUE_MAX_FENTRY         6000

/* Queue commands */
#define CMD_EMPTY                 10
#define CMD_CREATE                11
#define CMD_ISSUED                12
#define CMD_FINALIZE              13

/* Two-sided */
#define CMD_SEND                  14
#define CMD_RECV                  15
#define CMD_ISEND                 16
#define CMD_IRECV                 17
#define CMD_WAIT                  18
#define CMD_TEST                  19

/* One-sided */
#define CMD_WINCREATE             30
#define CMD_WINALLOCATE           31
#define CMD_PUT                   32
#define CMD_GET                   33
#define CMD_WINLOCK               34
#define CMD_WINUNLOCK             35
#define CMD_WINLOCKALL            36
#define CMD_WINUNLOCKALL          37
#define CMD_WINFLUSH              38
#define CMD_WINFLUSHLOCAL         39
#define CMD_WINFLUSHALL           40
#define CMD_WINFENCE              41
#define CMD_WINFREE               42

/* Collective calls */
#define CMD_BARRIER               50
#define CMD_IALLREDUCE            51
#define CMD_IALLTOALL             52
#define CMD_IALLGATHER            53
#define CMD_IREDUCE_SCATTER_BLOCK 54
#define CMD_IREDUCE               55
#define CMD_IBCAST                56
#define CMD_IGATHER               57

/* Atomics */
#define CMD_COMPARESWAP           60
#define CMD_FETCHOP               61

/* File I/O */
#define CMD_FOPEN                 70
#define CMD_FREAD                 71
#define CMD_FREAD_NB              72
#define CMD_FORC_NB               73
#define CMD_FCLOSE                74

/* Communicator operations */
#define CMD_COMM_SPLIT            80
#define CMD_COMM_CREATE_PEER      81
#define CMD_COMM_FREE             82
#define CMD_COMM_SET_INFO         83

/* Misc */
#define CMD_MEMORY_REGISTER       200
#define CMD_MEMORY_RELEASE        201
#define CMD_CONVERT_MPI_TYPES     202
#define CMD_CONVERT_MPI_OPS       203

/* Return values */
#define CMD_SUCCESS               1111
#define CMD_INFLIGHT              2222

/* Cqueue execution options */
#define CQUEUE_SUSPEND            0
#define CQUEUE_EXECUTE            1

struct __centry_t
{
    volatile int cmd;
    volatile void* buffer;
    volatile int count;
    volatile MPI_Datatype datatype;
    union
    {
        volatile int src_rank;
        volatile int count2;
        volatile int color;
        volatile int memid;
    };
    union
    {
        volatile int tgt_rank;
        volatile int root;
        volatile int key;
    };
    union
    {
        volatile int tag;
        volatile int inplace;
    };
    volatile MPI_Comm comm;
    volatile MPI_Op op;
    volatile void* buffer2;
    volatile MPI_Datatype datatype2;
    volatile size_t size_bytes;

#ifdef ENABLE_MPIRMA_ENDPOINTS
    volatile MPI_Win win;
    volatile void* buffer3;
    volatile MPI_Aint target_disp;
    volatile int target_count;
    volatile MPI_Datatype target_datatype;
    union
    {
        volatile int win_size;
        volatile int lock_type;
        volatile int test_flag;
    };
    union
    {
        volatile int disp_unit;
        volatile int assert;
    };
    MPI_Info info;
#endif /* ENABLE_MPIRMA_ENDPOINTS */

    volatile MPI_Status status;
    volatile int ret;
    long centryid;               /* Used only by the client */
    MPI_Request request;        /* Used only by the server */
    int is_prio_request;
} __attribute__ ((aligned (CACHELINE_SIZE)));

typedef struct __centry_t centry_t;

struct __fentry_t
{
    volatile int cmd;
    volatile void* buffer;
    volatile FILE* stream;
    volatile size_t size;
    volatile size_t count;
    volatile char filename[20];
    volatile char mode[6];
    volatile int ret;
    int fentryid;
} __attribute__ ((aligned (CACHELINE_SIZE)));

typedef struct __fentry_t fentry_t;

struct __cqueue_t
{
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned long head;
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned long tail;
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned int process;
    __attribute__ ((aligned(CACHELINE_SIZE))) centry_t centry_table[CQUEUE_MAX_CENTRY];
#ifdef ENABLE_FILEIO
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned long fhead;
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned long ftail;
    __attribute__ ((aligned(CACHELINE_SIZE))) volatile unsigned int fprocess;
    __attribute__ ((aligned(CACHELINE_SIZE))) fentry_t fentry_table[CQUEUE_MAX_FENTRY];
#endif /* ENABLE_FILEIO */
} __attribute__ ((aligned (CACHELINE_SIZE)));

typedef struct __cqueue_t cqueue_t;

void cqueue_init(int);
cqueue_t* cqueue_attach(int);
centry_t* cqueue_get_centry(long);
void cqueue_execute();
void cqueue_suspend();
int cqueue_send(cqueue_t*, const void*, int, MPI_Datatype, int, int, MPI_Comm);
int cqueue_recv(cqueue_t*, void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*);
int cqueue_isend(cqueue_t*, const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int cqueue_irecv(cqueue_t*, void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);
int cqueue_wait(centry_t*, request_t*, MPI_Status*);
int cqueue_test(centry_t*, request_t*, int*, MPI_Status*);
int cqueue_win_allocate(cqueue_t*, MPI_Aint, int, MPI_Info, MPI_Comm, void*, MPI_Win*);
int cqueue_win_create(cqueue_t*, void*, MPI_Aint, int, MPI_Info, MPI_Comm, MPI_Win*);
int cqueue_put(cqueue_t*, const void*, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
int cqueue_get(cqueue_t*, void*, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype, MPI_Win);
int cqueue_win_lock(cqueue_t*, int, int, int, MPI_Win);
int cqueue_win_unlock(cqueue_t*, int, MPI_Win);
int cqueue_win_lock_all(cqueue_t*, int, MPI_Win);
int cqueue_win_unlock_all(cqueue_t*, MPI_Win);
int cqueue_win_flush(cqueue_t*, int, MPI_Win);
int cqueue_win_flush_local(cqueue_t*, int, MPI_Win);
int cqueue_win_flush_all(cqueue_t*, MPI_Win);
int cqueue_win_fence(cqueue_t*, int, MPI_Win);
int cqueue_win_free(cqueue_t*, MPI_Win*);
int cqueue_barrier(cqueue_t*, MPI_Comm);
int cqueue_iallreduce(cqueue_t*, const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm, MPI_Request*);
int cqueue_ialltoall(cqueue_t*, const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm, MPI_Request*);
int cqueue_iallgather(cqueue_t*, const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm, MPI_Request*);
int cqueue_igather(cqueue_t*, const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm, MPI_Request*);
int cqueue_ireduce_scatter_block(cqueue_t*, const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm, MPI_Request*);
int cqueue_ibcast(cqueue_t*, void*, int, MPI_Datatype, int, MPI_Comm, MPI_Request*);
int cqueue_ireduce(cqueue_t*, const void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm, MPI_Request*);
int cqueue_compare_and_swap(cqueue_t*, const void*, const void*, void*, MPI_Datatype, int, MPI_Aint, MPI_Win);
int cqueue_fetch_and_op(cqueue_t*, const void*, void*, MPI_Datatype, int, MPI_Aint, MPI_Op, MPI_Win);
int cqueue_create_server_comm(MPI_Comm, int, int, MPI_Comm*, MPI_Comm*);
int cqueue_comm_free(MPI_Comm);
int cqueue_comm_set_info(cqueue_t*, MPI_Comm, const char*, const char*);
void cqueue_process(cqueue_t*, MPI_Comm);
int cqueue_memory_register(void*, size_t, int*);
int cqueue_memory_release(int);
int cqueue_mpi_type_register();
int cqueue_mpi_op_register();
void cqueue_finalize();

#ifdef ENABLE_FILEIO
fentry_t* cqueue_get_fentry(int);
int cqueue_fopen(cqueue_t*, const char*, const char*, FILE**);
int cqueue_fread_nb(cqueue_t*, void*, size_t, size_t, FILE*, MPI_Request*);
int cqueue_forc_nb(cqueue_t*, const char*, const char*, void*, size_t, size_t, MPI_Request*);
int cqueue_fwait(fentry_t*, request_t*, size_t*);
int cqueue_fclose(cqueue_t*, FILE*);
#endif /* ENABLE_FILEIO */

#endif /* _CQUEUE_H_ */
