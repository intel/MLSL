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
 * MPI Endpoints Communicator Handle Interface
 *
 */

#include "client.h"
#include "common.h"
#include "cqueue.h"
#include "debug.h"
#include "handle.h"
#include "env.h"
#include "types.h"

#define EP_HANDLE_TABLE_INC 256

handle_t* handle_table;
long handle_table_size = 0;

static void handle_table_grow(long inc)
{
    long i;

    if (handle_table == NULL)
        MALLOC_ALIGN(handle_table, (handle_table_size+inc) * sizeof(handle_t), CACHELINE_SIZE);
    else
    {
        handle_t* handle_table_new;
        MALLOC_ALIGN(handle_table_new, (handle_table_size+inc) * sizeof(handle_t), CACHELINE_SIZE);
        memcpy(handle_table_new, handle_table, handle_table_size * sizeof(handle_t));
        FREE_ALIGN(handle_table);
        handle_table = handle_table_new;
    }

    ASSERT(handle_table != NULL);

    for (i = handle_table_size; i < handle_table_size + inc; i++)
    {
        ASSERT_FMT((intptr_t) &handle_table[i] % CACHELINE_SIZE == 0,
                  "&handle_table[%ld] %% CACHELINE_SIZE == 0", i);
        handle_table[i].in_use = 0;
        handle_table[i].worldcomm = NULL;
        handle_table[i].peercomm = NULL;
    }

    handle_table_size += inc;

    if ((long) MPI_COMM_WORLD >= 0 && (long) MPI_COMM_WORLD < handle_table_size)
        handle_table[(long) MPI_COMM_WORLD].in_use = 1;
    if ((long) MPI_COMM_SELF >= 0 && (long) MPI_COMM_SELF < handle_table_size)
        handle_table[(long) MPI_COMM_SELF].in_use = 1;
    if ((long) MPI_COMM_NULL >= 0 && (long) MPI_COMM_NULL < handle_table_size)
        handle_table[(long) MPI_COMM_NULL].in_use = 1;
}

void handle_init()
{
    handle_table_grow(EP_HANDLE_TABLE_INC);
}

long handle_register(int type, MPI_Comm comm, MPI_Comm parent_comm, int num_endpoints,
                    int local_epid, int tot_endpoints, int global_epid,
                    MPI_Comm* worldcomm, MPI_Comm* peercomm)
{
    long hdl;

    if ((long) comm >= 0 && (long) comm < handle_table_size)
    ERROR("Conflict in communicator values between client and server.\n");

    for (hdl = 0; hdl < handle_table_size; hdl++)
        if (!handle_table[hdl].in_use) break;

    if (hdl == handle_table_size)
        handle_table_grow(EP_HANDLE_TABLE_INC);

    handle_table[hdl].in_use = 1;
    handle_table[hdl].type = type;
    handle_table[hdl].parent_comm = parent_comm;
    handle_table[hdl].num_endpoints = num_endpoints;
    handle_table[hdl].local_epid = local_epid;
    handle_table[hdl].tot_endpoints = tot_endpoints;
    handle_table[hdl].global_epid = global_epid;

    if (type == COMM_REG)
    {
        handle_table[hdl].comm = comm;
        handle_table[hdl].cqueue = NULL;
        if (num_endpoints > 0)
        {
	    if (std_mpi_mode == STD_MPI_MODE_EXPLICIT)
            {
		if (comm == parent_comm)
                {
		    /* Primitive regular communicator like MPI_COMM_WORLD */
		    handle_table[hdl].cqueue = client_get_cqueue(handle_table[hdl].local_epid);
		    handle_table[hdl].local_epid = (handle_table[hdl].local_epid + 1) % num_endpoints;
		}
		else
		{
		    /* Regular communicator created via comm_split or comm_dup */
		    /* Find the parent communicator's entry and get the local epid */
		    for (long thdl = 0; thdl < handle_table_size; thdl++)
		    {
			if (handle_table[thdl].comm == parent_comm && handle_table[thdl].type == COMM_REG)
			{
			    handle_table[hdl].cqueue = client_get_cqueue(handle_table[thdl].local_epid);
			    handle_table[thdl].local_epid = (handle_table[thdl].local_epid + 1) % num_endpoints;
			    handle_table[hdl].local_epid = handle_table[thdl].local_epid;
			}
		    }
		}
	    }

            MALLOC_ALIGN(handle_table[hdl].worldcomm, num_endpoints * sizeof(MPI_Comm), CACHELINE_SIZE);
            MALLOC_ALIGN(handle_table[hdl].peercomm, num_endpoints * sizeof(MPI_Comm), CACHELINE_SIZE);
            for (int i = 0; i < num_endpoints; i++)
            {
                handle_table[hdl].worldcomm[i] = worldcomm[i];
                handle_table[hdl].peercomm[i] = peercomm[i];
            }
        }
    }
    else
    {
        handle_table[hdl].comm = (MPI_Comm)hdl;
        handle_table[hdl].cqueue = client_get_cqueue(local_epid);
        MALLOC_ALIGN(handle_table[hdl].worldcomm, sizeof(MPI_Comm), CACHELINE_SIZE);
        MALLOC_ALIGN(handle_table[hdl].peercomm, sizeof(MPI_Comm), CACHELINE_SIZE);
        for (long thdl = 0; thdl < handle_table_size; thdl++)
        {
            if (handle_table[thdl].comm == parent_comm && handle_table[thdl].type == COMM_REG)
            {
                handle_table[hdl].worldcomm[0] = handle_table[thdl].worldcomm[local_epid];
                handle_table[hdl].peercomm[0] = handle_table[thdl].peercomm[local_epid];
                break;
            }
            else if (handle_table[thdl].comm == (MPI_Comm)parent_comm && handle_table[thdl].type == COMM_EP)
                ERROR("Cannot use EP comm as parent comm in MPI_Comm_create_endpoints");
        }
    }
    return hdl;
}

int handle_get_type(MPI_Comm comm)
{
    if ((long) comm >= 0 && (long) comm < handle_table_size)
    {
        DEBUG_ASSERT(handle_table[(long)comm].in_use);
        return COMM_EP;
    }
    return COMM_REG;
}

int handle_get_rank(MPI_Comm hdl)
{
    DEBUG_ASSERT(handle_table[(long)hdl].in_use);
    return handle_table[(long)hdl].global_epid;
}

int handle_get_size(MPI_Comm hdl)
{
    DEBUG_ASSERT(handle_table[(long)hdl].in_use);
    return handle_table[(long)hdl].tot_endpoints;
}

inline cqueue_t* handle_get_cqueue(MPI_Comm comm)
{
    if ((long) comm >= 0 && (long) comm < handle_table_size)
    {
        /* Endpoints communicator */
        DEBUG_ASSERT(handle_table[(long)comm].in_use);
        return handle_table[(long)comm].cqueue;
    }
    else
    {
	/* Regular communicator */
	if (std_mpi_mode == STD_MPI_MODE_EXPLICIT)
	{
	  for (long thdl = 0; thdl < handle_table_size; thdl++)
            if (handle_table[thdl].in_use && handle_table[thdl].comm == comm && handle_table[thdl].type == COMM_REG)
                return handle_table[(long)thdl].cqueue;
	}
        return NULL;
    }
}

inline MPI_Comm handle_get_server_comm(MPI_Comm comm, int epid)
{
    if ((long) comm >= 0 && (long) comm < handle_table_size)
    {
        DEBUG_ASSERT(handle_table[(long)comm].in_use);
#ifdef ENABLE_PEERCOMM
        return handle_table[(long) comm].peercomm[0];
#else
	return handle_table[(long) comm].worldcomm[0];
#endif
    }
    else
        return handle_get_server_peercomm(comm, epid);
}

MPI_Comm handle_get_server_worldcomm(MPI_Comm comm, int epid)
{
    for (long thdl = 0; thdl < handle_table_size; thdl++)
        if (handle_table[thdl].in_use && handle_table[thdl].comm == comm && handle_table[thdl].type == COMM_REG)
            return handle_table[thdl].worldcomm[epid];
    return EPLIB_COMM_NULL;
}

MPI_Comm handle_get_server_peercomm(MPI_Comm comm, int epid)
{
    for (long thdl = 0; thdl < handle_table_size; thdl++)
        if (handle_table[thdl].in_use && handle_table[thdl].comm == comm && handle_table[thdl].type == COMM_REG)
            return handle_table[thdl].peercomm[epid];
    return EPLIB_COMM_NULL;
}

void handle_release(MPI_Comm comm)
{
    if (!handle_table) return;

    for (long thdl = 0; thdl < handle_table_size; thdl++)
    {
        if (!handle_table[thdl].in_use) continue;

        if (handle_table[thdl].comm == comm)
        {
            if (handle_table[thdl].peercomm != NULL)
            {
                if (handle_table[thdl].type == COMM_REG)
                    cqueue_comm_free(handle_table[thdl].comm);
                FREE_ALIGN(handle_table[thdl].peercomm);
            }
            if (handle_table[thdl].worldcomm != NULL)
                FREE_ALIGN(handle_table[thdl].worldcomm);
            handle_table[thdl].in_use = 0;
        }
    }
}

void handle_finalize()
{
    if (!handle_table) return;

    for (long thdl = 0; thdl < handle_table_size; thdl++)
      handle_release((MPI_Comm)thdl);
    FREE_ALIGN(handle_table);
    handle_table = NULL;
}
