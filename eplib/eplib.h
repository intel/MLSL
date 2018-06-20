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
#ifndef _WRAPPER_H_
#define _WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

    int MPI_Comm_create_endpoints(MPI_Comm, int, MPI_Info, MPI_Comm []);

    void EPLIB_split_comm(MPI_Comm parentcomm, int color, int key, MPI_Comm newcomm);
    int EPLIB_comm_set_info(MPI_Comm* comms, size_t comm_count, const char* key, const char* value);

    /* Init / teardown interface */
    int EPLIB_init();
    int EPLIB_finalize(void);

    /* Memory management interface */
    void* EPLIB_malloc(size_t);
    void* EPLIB_realloc(void*, size_t);
    void* EPLIB_calloc(size_t, size_t);
    void* EPLIB_memalign(size_t, size_t);
    void EPLIB_free(void*);
    int EPLIB_memory_is_shmem(void*);
    void EPLIB_set_mem_hooks();
    void* EPLIB_quant_params_submit(void*);

    /* Endpoint server management interface */
    void EPLIB_execute();
    void EPLIB_suspend();

    /* File operations */
    FILE* EPLIB_fopen(int, const char*, const char*);
    size_t EPLIB_fread(int, void*, size_t, size_t, FILE*);
    size_t EPLIB_fread_nb(int, void*, size_t, size_t, FILE*, MPI_Request*);
    size_t EPLIB_forc_nb(int, const char*, const char*, void*, size_t, size_t, MPI_Request*);
    int EPLIB_fwait(MPI_Request*, size_t*);
    int EPLIB_fwaitall(int, MPI_Request*, size_t*);
    int EPLIB_fclose(int, FILE*);

#ifdef __cplusplus
}
#endif

#endif /* _WRAPPER_H_ */
