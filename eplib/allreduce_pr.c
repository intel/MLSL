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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "allreduce_pr.h"
#include "cqueue.h"
#include "env.h"

int local_size;
int local_rank;
int ghead, tail = 0;
int tag_ub = 32767;

allreduce_pr_req allrdc_req[MAX_ALLREDUCE_REQS];
allreduce_pr_buf allrdc_buf[MAX_ALLREDUCE_REQS];

void* allreduce_pr_get_buf(size_t size)
{
    for (int i = 0; i < MAX_ALLREDUCE_REQS; i++)
    {
        if (allrdc_buf[i].size == size && !allrdc_buf[i].in_use)
        {
            allrdc_buf[i].in_use = 1;
            return allrdc_buf[i].buf;
        }
    }
    for (int i = 0; i < MAX_ALLREDUCE_REQS; i++)
    {
        if (!allrdc_buf[i].buf && size > 0)
        {
            MALLOC_ALIGN(allrdc_buf[i].buf, size, 4096);
            allrdc_buf[i].in_use = 1;
            allrdc_buf[i].size = size;
            return allrdc_buf[i].buf;
        }
    }
    return NULL;
}

void allreduce_pr_release_buf(void* buf)
{
    for (int i = 0; i < MAX_ALLREDUCE_REQS; i++) 
    {
        if (allrdc_buf[i].buf == buf && allrdc_buf[i].in_use)
        {
            allrdc_buf[i].in_use = 0;
            break;
        }
    }
}

void allreduce_pr_make_progress(int req_idx, int progress_req)
{
    int flag;
    for (int m = 0; m < MAX_ALLREDUCE_REQS; m++)
    {
        int i;

        if (msg_priority_mode)
            i = (ghead - m + MAX_ALLREDUCE_REQS) % MAX_ALLREDUCE_REQS;
        else
            i = (tail + m + MAX_ALLREDUCE_REQS) % MAX_ALLREDUCE_REQS;

        if (progress_req) i = req_idx;

        if (allrdc_req[i].state == REQ_FREE || allrdc_req[i].state == REQ_COMPLETED) continue;

        const void* sendbuf = allrdc_req[i].sendbuf;
        void* recvbuf       = allrdc_req[i].recvbuf;
        void* src;
        void* dst = NULL;
        int count = allrdc_req[i].count;
        int tag   = allrdc_req[i].tag;
        MPI_Datatype datatype = allrdc_req[i].datatype;
        MPI_Aint extent, lb;
        MPI_Type_get_extent(datatype, &lb, &extent);
        MPI_Status status;

        int num_of_steps = log2(local_size);
        int cur_pos = 0;

        DEBUG_PRINT("[%d]:  i=%d phase %d pos %d size=%d steps=%d extent=%d %p %p\n",
                    local_rank, i, allrdc_req[i].phase, cur_pos, local_size, num_of_steps, (int)extent, sendbuf, recvbuf);

        for (int j = 0; j < num_of_steps; j++)
        {
            int ex_size = count / pow(2, j+1);
            int pair;
            int jump = pow(2, j);
            int pair_dir = (local_rank / jump) % 2;
            if (pair_dir)
                pair = local_rank - jump;
            else
            {
                pair = local_rank + jump;
                cur_pos += ex_size;
            }
            if (j == 0)
            {
                dst = (char*)recvbuf + cur_pos * extent;
                if (!allrdc_req[i].phase && sendbuf == MPI_IN_PLACE)
                {
                    dst = allreduce_pr_get_buf(ex_size * extent);
                    allrdc_req[i].inbuf = dst;
                } else if (allrdc_req[i].phase && sendbuf == MPI_IN_PLACE)
                    dst = allrdc_req[i].inbuf;
            }

            if (j < allrdc_req[i].phase / 2)
            {
                if (pair_dir)
                    cur_pos += ex_size;
                else
                    cur_pos -= ex_size;
                continue;
            }

            if (allrdc_req[i].phase % 2) goto check_completion;

            src = (char*)recvbuf + cur_pos * extent;
            if (j == 0)
            {
                PMPI_Irecv(dst, ex_size, datatype, pair, tag, allrdc_req[i].comm, &allrdc_req[i].req[0]);
                if (sendbuf == MPI_IN_PLACE)
                    PMPI_Isend((char*)recvbuf + cur_pos * extent, ex_size, datatype, pair, tag, allrdc_req[i].comm, &allrdc_req[i].req[1]);
                else
                {
                    PMPI_Isend((char*)sendbuf + cur_pos * extent, ex_size, datatype, pair, tag, allrdc_req[i].comm, &allrdc_req[i].req[1]);
                    if (datatype == MPI_DOUBLE)
                    {
                        double* src = (double*)((char*)sendbuf + ((cur_pos + ex_size) % count) * extent);
                        double* dst = (double*)((char*)recvbuf + ((cur_pos + ex_size) % count) * extent);
                        for (int i = 0; i < ex_size; i++)
                            dst[i] = src[i];
                    }
                    if (datatype == MPI_FLOAT)
                    {
                        float* src = (float*)((char*)sendbuf + ((cur_pos + ex_size) % count) * extent);
                        float* dst = (float*)((char*)recvbuf + ((cur_pos + ex_size) % count) * extent);
                        for (int i = 0; i < ex_size; i++)
                            dst[i] = src[i];
                    }
                    if (datatype == MPI_CHAR)
                    {
                        char* src = (char*)sendbuf + ((cur_pos + ex_size) % count) * extent;
                        char* dst = (char*)recvbuf + ((cur_pos + ex_size) % count) * extent;

                        for (int i = 0; i < ex_size; i++)
                            dst[i] = src[i];
                    }
                }
            }
            else
            {
                PMPI_Isend(src, ex_size, datatype, pair, tag + j, allrdc_req[i].comm, &allrdc_req[i].req[1]);
                PMPI_Irecv(dst, ex_size, datatype, pair, tag + j, allrdc_req[i].comm, &allrdc_req[i].req[0]);
            }
            allrdc_req[i].phase++;

            ASSERT(dst != NULL);

check_completion:
            flag = 0;
            PMPI_Testall(2, &allrdc_req[i].req[0], &flag, &status);

            if (!flag) goto next_request;

            allrdc_req[i].phase++;

            DEBUG_PRINT("[%d]: phase %d pos %d\n", local_rank, allrdc_req[i].phase, cur_pos);

            if (pair_dir)
                cur_pos += ex_size;
            else
                cur_pos -= ex_size;
            src = (char*)recvbuf + cur_pos * extent;
            if (datatype == MPI_DOUBLE)
            {
                double* lsrc = (double*)src;
                double* ldst = (double*)dst;
                for (int k = 0; k < ex_size; k++)
                    lsrc[k] += ldst[k];
            }
            if (datatype == MPI_FLOAT)
            {
                float* lsrc = (float*)src;
                float* ldst = (float*)dst;
                for (int k = 0; k < ex_size; k++)
                    lsrc[k] += ldst[k];
            }
            if (datatype == MPI_CHAR)
            {
                char* lsrc = (char*)src;
                char* ldst = (char*)dst;

                for (int k = 0; k < ex_size; k++)
                    lsrc[k] += ldst[k];
            }
        }

        DEBUG_PRINT("[%d]: phase %d pos %d\n", local_rank, allrdc_req[i].phase, cur_pos);

        if (sendbuf == MPI_IN_PLACE && allrdc_req[i].inbuf)
        {
            allreduce_pr_release_buf(allrdc_req[i].inbuf);
            allrdc_req[i].inbuf = NULL;
        }

        for (int j = num_of_steps - 1; j >= 0; j--)
        {
            int ex_size = allrdc_req[i].count / pow(2, j + 1);
            int pair;
            int jump = pow(2, j);
            int send_pos = cur_pos;
            int dst_pos = cur_pos;
            int pair_dir = (local_rank / jump) % 2;
            if (pair_dir)
            {
                pair = local_rank - jump;
                dst_pos -= ex_size;
            }
            else
            {
                pair = local_rank + jump;
                dst_pos += ex_size;
            }
            if (num_of_steps - j - 1 != (allrdc_req[i].phase - 2 * num_of_steps) / 2)
            {
                if(pair_dir)
                    cur_pos -= ex_size;
                continue;
            }
            if (allrdc_req[i].phase % 2 == 0)
            {
                src = (char*)recvbuf + send_pos * extent;
                dst = (char*)recvbuf +  dst_pos * extent;

                DEBUG_PRINT("[%d]: send %d dst %d\n", local_rank, send_pos, dst_pos);

                PMPI_Irecv(dst, ex_size, datatype, pair, tag + num_of_steps + j, allrdc_req[i].comm, &allrdc_req[i].req[0]);
                PMPI_Isend(src, ex_size, datatype, pair, tag + num_of_steps + j, allrdc_req[i].comm, &allrdc_req[i].req[1]);
                allrdc_req[i].phase++;
            }
            int flag=0;
            PMPI_Testall(2, &allrdc_req[i].req[0], &flag, &status);

            if (!flag) goto next_request;

            allrdc_req[i].phase++;
            if (pair_dir)
                cur_pos -= ex_size;
        }

        allrdc_req[i].state = REQ_COMPLETED;
        tail = (tail + 1) % MAX_ALLREDUCE_REQS;
        continue;
next_request:
        flag = 0;
        return;
    }
}

int allreduce_pr_is_done(int req_idx)
{
    int ret = 0;
    allreduce_pr_make_progress(req_idx, 0);
    if (allrdc_req[req_idx].state == REQ_COMPLETED)
    {
        ret = 1;
        allrdc_req[req_idx].state = REQ_FREE;
    }
    return ret;
}

void allreduce_pr_initialize()
{
    PMPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &local_size);
    for (int i = 0; i < MAX_ALLREDUCE_REQS; i++)
        allrdc_req[i].state = REQ_FREE;

    int flag;
    int* tag_ub_ptr;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
    if (flag) tag_ub = *tag_ub_ptr;
}

int allreduce_pr_start(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm, MPI_Request* request)
{
    static int tag = 0;
    if (!tag)
    {
        PMPI_Comm_rank(comm, &local_rank);
        PMPI_Comm_size(comm, &local_size);
        for (int i = 0; i < MAX_ALLREDUCE_REQS; i++)
            allrdc_req[i].state = REQ_FREE;
    }

    tag += 4 * local_size;
    if (tag > tag_ub) tag = 4 * local_size;
    ASSERT(tag <= tag_ub);

    int i = ghead;
    ghead = (ghead + 1) % MAX_ALLREDUCE_REQS;

    if (allrdc_req[i].state == REQ_FREE)
    {
        allrdc_req[i].sendbuf  = sendbuf;
        allrdc_req[i].recvbuf  = recvbuf;
        allrdc_req[i].count    = count;
        allrdc_req[i].op       = op;
        allrdc_req[i].comm     = comm;
        allrdc_req[i].datatype = datatype;
        allrdc_req[i].phase    = 0;
        allrdc_req[i].inbuf    = NULL;
        allrdc_req[i].tag      = tag;
        allrdc_req[i].state    = REQ_IN_PROGRESS;
        *(int*)request         = i;
        allreduce_pr_make_progress(i, 1);
        return MPI_SUCCESS;
    }
    ERROR("[%d]: CANNOT FIND EMPTY ALLREDUCE SLOT \n", local_rank);
}

void allreduce_pr_test(MPI_Request* req, int* flag)
{
    int req_idx = *(int*)req;
    allreduce_pr_make_progress(req_idx, 0);
    if (allrdc_req[req_idx].state == REQ_COMPLETED)
    {
        *flag = 1;
        allrdc_req[req_idx].state = REQ_FREE;
    } else
        *flag = 0;
}
