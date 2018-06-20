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
#include <stdio.h> /* popen */
#include <stdlib.h> /* strtoull */

#include "common.h"
#include "env.h"
#include "memory.h"
#include "cqueue.h"

#include "debug.h"
#include "dlmalloc.c"

#define EP_MEMORY_TABLE_INC 16

#define SHM_FILENAME_PREFIX   "eplib_shm."
#define MEMORY_TABLE_MAX_SIZE 99
#define MEMORY_TABLE_IDX_LEN  sizeof(STRINGIFY(MEMORY_TABLE_MAX_SIZE))
#define SHM_FILENAME_LEN      (sizeof(SHM_FILENAME_PREFIX) + UUID_STR_LEN + MEMORY_TABLE_IDX_LEN)

struct __memory_t
{
    int in_use;
    int do_create;
    void* shm_base;
    void* client_shm_base;
    mspace shm_mspace;
    int shm_id;
    size_t mem_size;
    char shm_filename[SHM_FILENAME_LEN];
} __attribute__ ((aligned (CACHELINE_SIZE)));

typedef struct __memory_t memory_t;

memory_t* memory_table;
int memory_table_size = 0;
int memory_table_last = 0;
int need_unlink = 1;

size_t max_memory_size = 0;
size_t max_shm_memory_size = 0;
size_t allocated_memory_size = 0;

static void memory_table_grow(int inc)
{
    if (memory_table_size + inc > MEMORY_TABLE_MAX_SIZE)
        ERROR("Max limit on # memory regions is %d. Increase EPLIB_SHM_SIZE_GB.\n",
              MEMORY_TABLE_MAX_SIZE);

    if (memory_table == NULL)
        MALLOC_ALIGN(memory_table, (memory_table_size + inc) * sizeof(memory_t), CACHELINE_SIZE);
    else
    {
        memory_t* memory_table_new;
        MALLOC_ALIGN(memory_table_new, (memory_table_size + inc) * sizeof(memory_t), CACHELINE_SIZE);
        memcpy(memory_table_new, memory_table, memory_table_size * sizeof(memory_t));
        FREE_ALIGN(memory_table);
        memory_table = memory_table_new;
    }

    ASSERT(memory_table != NULL);

    for (int i = memory_table_size; i < memory_table_size + inc; i++)
    {
        ASSERT_FMT((intptr_t) &memory_table[i] % CACHELINE_SIZE == 0,
                  "&memory_table[%d] %% CACHELINE_SIZE == 0", i);
        memory_table[i].in_use = 0;
    }

    memory_table_size += inc;
}

#ifdef ENABLE_CLIENT_ONLY
void memory_get_limits()
{
    if (!check_mem_size) return;

    max_memory_size = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);

    ASSERT(sizeof(size_t) == sizeof(unsigned long long int));
    char shm_size_cmd[1024] = "df --block-size=1 /dev/shm | tail -n 1 | awk '{ print $2 }'";
    char shm_size_str[1024];
    FILE* pipe = popen(shm_size_cmd, "r");
    ASSERT(pipe);
    ASSERT(fgets(shm_size_str, sizeof(shm_size_str), pipe));
    max_shm_memory_size = strtoull(shm_size_str, NULL, 10);
    pclose(pipe);

    DEBUG_PRINT("memory_get_limits: ram size: %zu, shm size: %zu\n",
                max_memory_size, max_shm_memory_size);

    ASSERT(max_memory_size > 0);
    ASSERT(max_shm_memory_size > 0);
}

size_t memory_get_free_ram_size()
{
    char ram_size_cmd[1024] = "free -b | grep 'Mem:' | awk '{print $4}'";
    char ram_size_str[1024];
    FILE* pipe = popen(ram_size_cmd, "r");
    ASSERT(pipe);
    ASSERT(fgets(ram_size_str, sizeof(ram_size_str), pipe));
    size_t free_ram_size = strtoull(ram_size_str, NULL, 10);
    pclose(pipe);

    DEBUG_PRINT("memory_get_free_ram_size: ram size: %zu\n",
                free_ram_size);
    return free_ram_size;
}

size_t memory_get_free_shm_size()
{
    char shm_size_cmd[1024] = "df --block-size=1 /dev/shm | tail -n 1 | awk '{ print $4 }'";
    char shm_size_str[1024];
    FILE* pipe = popen(shm_size_cmd, "r");
    ASSERT(pipe);
    ASSERT(fgets(shm_size_str, sizeof(shm_size_str), pipe));
    size_t free_shm_size = strtoull(shm_size_str, NULL, 10);
    pclose(pipe);

    DEBUG_PRINT("memory_get_free_shm_size: shm size: %zu\n",
                free_shm_size);
    return free_shm_size;
}
#endif

void memory_init()
{
    if (!use_allocator) return;

    memory_table_grow(EP_MEMORY_TABLE_INC);
}

int memory_register(void* baseaddr, size_t mem_size, const char* uuid, int do_create)
{
    if (!use_allocator) return 0;

    int memid;
    size_t offset = 0;
    size_t max_quant_params_count = 1;

    for (memid = 0; memid < memory_table_size; memid++)
        if (!memory_table[memid].in_use) break;

    if (memid == memory_table_size)
        memory_table_grow(EP_MEMORY_TABLE_INC);

    memory_t* memptr = &memory_table[memid];
    memptr->in_use = 1;
    memptr->do_create = do_create;
    memptr->shm_base = NULL;
    memptr->client_shm_base = baseaddr;
    memptr->shm_mspace = NULL;
    memptr->shm_id = -1;
    memptr->mem_size = mem_size;

    if (baseaddr == NULL)
        offset = (((sizeof(intptr_t*) * (max_ep + 1)) / PAGE_SIZE) + 1) * PAGE_SIZE;

    if (do_create)
    {
        /* Client */
        /* Create a shared memory segment */
        snprintf(memptr->shm_filename, SHM_FILENAME_LEN, SHM_FILENAME_PREFIX"%s%d", uuid, memid);
        if ((memptr->shm_id = shm_open(memptr->shm_filename, O_RDWR|O_CREAT, S_IRWXU|S_IRWXG|S_IROTH)) < 0)
        {
            PRINT("CLIENT: shm_open failed (%s)\n", strerror(errno));
            memory_release(memid);
            return -1;
        }

        /* Adjust the size of the shared memory segment */
        if (ftruncate(memptr->shm_id, mem_size) < 0)
        {
            PRINT("CLIENT: ftruncate failed (%s)\n", strerror(errno));
            memory_release(memid);
            return -1;
        }

        /* Create mmap region */
        if ((memptr->shm_base = (void*)mmap(baseaddr, mem_size, (PROT_WRITE|PROT_READ),
                                            (baseaddr) ? (MAP_FIXED|MAP_SHARED) : MAP_SHARED,
                                            memptr->shm_id, 0)) == MAP_FAILED)
        {
            PRINT("CLIENT: mmap failed (%s)\n", strerror(errno));
            memory_release(memid);
            return -1;
        }
    }
    else
    {
        /* Server */
        /* Open shared memory region */
        snprintf(memptr->shm_filename, SHM_FILENAME_LEN, SHM_FILENAME_PREFIX"%s%d", uuid, memid);
        if ((memptr->shm_id = shm_open(memptr->shm_filename, O_RDWR, S_IRWXU|S_IRWXG|S_IROTH)) < 0)
        {
            DEBUG_PRINT("SERVER: shm_open failed (%s)\n", strerror(errno));
            memory_release(memid);
            return -1;
        }

        /* Create mmap region */
        if ((memptr->shm_base = (void*)mmap(NULL, mem_size, PROT_WRITE|PROT_READ, MAP_SHARED, memptr->shm_id, 0)) == MAP_FAILED)
        {
            DEBUG_PRINT("SERVER: mmap failed (%s)\n", strerror(errno));
            memory_release(memid);
            return -1;
        }
    }

    ASSERT(memptr->shm_base != NULL);

    if (do_create)
    {
        memptr->client_shm_base = memptr->shm_base;
        if (baseaddr == NULL)
        {
            /* Track client shmem address and server cqueue address */
            for (int i = 0; i < max_ep + 1 + max_quant_params_count; i++)
            {
                void* cqueue_base = (intptr_t*)memptr->shm_base + i;
                if (i == 0)
                    /* Index 0: client shared memory region */
                    *(intptr_t*)cqueue_base = (intptr_t)memptr->shm_base;
                else
                    /* Index 1 to max_ep: server cqueue address */
                    *(intptr_t*)cqueue_base = (intptr_t)-1;
            }
            /* Create dlmalloc mspace only for allocated regions */
            intptr_t mspace_start = (intptr_t)memptr->shm_base + (intptr_t)offset;
            memptr->shm_mspace = create_mspace_with_base((void*)mspace_start, mem_size-offset, 1);
            if (memptr->shm_mspace == NULL)
            {
                memory_release(memid);
                return -1;
            }
        }
        DEBUG_PRINT("CLIENT: %p %ld %ld %p %ld %p\n",
                    memptr->shm_base, *(intptr_t*)memptr->shm_base,
                    offset, memptr->shm_base + (intptr_t)offset,
                    mem_size, memptr->shm_mspace);
    }
    else
        DEBUG_PRINT("SERVER: %p %ld %ld %p %ld\n",
                    memptr->shm_base, *(intptr_t*)memptr->shm_base,
                    offset, memptr->shm_base + (intptr_t)offset,
                    mem_size);

    return memid;
}

void memory_set_cqueue(int cqueueid, void* mycqueue)
{
    if (!use_allocator) return;

    memory_t* memptr = &memory_table[0];
    ASSERT(memptr->in_use);
    void* cqueue_base = (intptr_t*)memptr->shm_base + cqueueid + 1;
    *(intptr_t*)cqueue_base = ((intptr_t)mycqueue - (intptr_t)memptr->shm_base);
}

void* memory_get_cqueue(int cqueueid)
{
    if (!use_allocator) return NULL;

    memory_t* memptr = &memory_table[0];
    ASSERT(memptr->in_use);
    void* cqueue_base = (intptr_t*)memptr->shm_base + cqueueid + 1;
    void* volatile* cqueue_baseptr = (void**)cqueue_base;
    while (*(intptr_t*)cqueue_baseptr == (intptr_t) - 1);
    return (void*)((intptr_t)*cqueue_baseptr + (intptr_t)memptr->shm_base);
}

void memory_set_quant_params(int quant_params_id, void* quant_params)
{
    if (!use_allocator) return;

    memory_t* memptr = &memory_table[0];
    ASSERT(memptr->in_use);
    void* cqueue_base = (intptr_t*)memptr->shm_base + max_ep + quant_params_id + 1;
    *(intptr_t*)cqueue_base = ((intptr_t)quant_params - (intptr_t)memptr->shm_base);
}

void* memory_get_quant_params(int quant_params_id)
{
    if (!use_allocator) return NULL;

    memory_t* memptr = &memory_table[0];
    ASSERT(memptr->in_use);
    void* cqueue_base = (intptr_t*)memptr->shm_base + max_ep + quant_params_id + 1;
    void* volatile* cqueue_baseptr = (void**)cqueue_base;
    while (*(intptr_t*)cqueue_baseptr == (intptr_t) - 1);
    return (void*)((intptr_t)*cqueue_baseptr + (intptr_t)memptr->shm_base);
}

void memory_get_client_shm_base(int memid)
{
    if (!use_allocator) return;

    memory_t* memptr = &memory_table[memid];
    ASSERT(memptr->in_use);
    if (memptr->client_shm_base == NULL)
        memptr->client_shm_base = (void*)(*(intptr_t*)memptr->shm_base);
}

inline int memory_is_shmem(void* mem, int* mem_idx)
{
    if (!use_allocator) return 1;

    if (mem_idx)
        *mem_idx = -1;

    for (int i = 0; i < memory_table_size; i++)
    {
        memory_t* memptr = &memory_table[i];
        if (memptr->in_use && mem >= memptr->client_shm_base && mem <= memptr->client_shm_base + memptr->mem_size)
        {
            if (mem_idx)
                *mem_idx = i;
            return 1;
        }
    }
    return 0;
}

inline void* memory_translate_clientaddr(void* mem)
{
    if (!use_allocator || !mem) return mem;

    int mem_idx;
    if (memory_is_shmem(mem, &mem_idx))
    {
        ASSERT((mem_idx >= 0 && mem_idx < memory_table_size));
        memory_t* memptr = &memory_table[mem_idx];
        DEBUG_ASSERT(memptr->in_use);
        return (void*)((intptr_t)mem - (intptr_t)memptr->client_shm_base + (intptr_t)memptr->shm_base);
    }

    ERROR("Message buffer (%p) not allocated by EPLIB\n", mem);
    return NULL;
}

#ifdef ENABLE_CLIENT_ONLY

static void memory_check_limits(size_t increment, int is_shm)
{
    if (!check_mem_size) return;
    allocated_memory_size += increment;

    DEBUG_PRINT("memory_check_limits: allocated_memory_size %zu, ram %zu, shm %zu\n",
                allocated_memory_size, max_memory_size, max_shm_memory_size);

    if (is_shm)
    {
        size_t free_shm_size = memory_get_free_shm_size();

        if ((double)free_shm_size <= (double)max_shm_memory_size * 0.1)
            PRINT("WARNING: you are close to /dev/shm limit (sizes in bytes: maximum size of /dev/shm: %zu, available size in /dev/shm: %zu)\n",
                  max_shm_memory_size, free_shm_size);

        if (increment > free_shm_size)
            ERROR("ERROR: not enough memory in /dev/shm: increase size of /dev/shm or delete stalled files from /dev/shm "
                  "(sizes in bytes: totally allocated: %zu, maximum size of /dev/shm: %zu, current increment: %zu, available size in /dev/shm: %zu). "
                  "Exit ...\n",
                   allocated_memory_size, max_shm_memory_size, increment, free_shm_size);
    }
    else
    {
        size_t free_ram_size = memory_get_free_ram_size();

        if ((double)free_ram_size <= (double)max_memory_size * 0.1)
            PRINT("WARNING: you are close to RAM size limit (sizes in bytes: maximum size of RAM: %zu, available size in RAM: %zu)\n",
                  max_memory_size, free_ram_size);

        if (increment > free_ram_size)
            ERROR("ERROR: not enough memory in RAM "
                  "(sizes in bytes: totally allocated: %zu, maximum size of RAM: %zu, current increment: %zu, available size in RAM: %zu). "
                  "Exit ...\n",
                   allocated_memory_size, max_memory_size, increment, free_ram_size);
    }
}

void* memory_expand(size_t bytes)
{
    int memid;
    size_t new_size = shm_size;
    while (new_size < bytes)
        new_size *= 2;

    memory_check_limits(new_size, 1);

    cqueue_memory_register(NULL, new_size, &memid);
    if (memid > memory_table_last)
        memory_table_last = memid;

    return &memory_table[memid];
}

void* memory_malloc(size_t bytes)
{
    void* mem = NULL;
    int i = 0;

    if (!use_allocator)
    {
        memory_check_limits(bytes, 0);
        MALLOC_ALIGN(mem, bytes, CACHELINE_SIZE);
        return mem;
    }

    while (mem == NULL)
    {
        memory_t* memptr = &memory_table[i];
        /* First try to allocate in one of the existing chunks */
        if (memptr->in_use == 1)
            if ((mem = mspace_malloc(memptr->shm_mspace, bytes)) != NULL)
                break;

        /* Create a new chunck if already past the last valid chunk */
        if (i > memory_table_last)
        {
            memptr = memory_expand(bytes);
            if ((mem = mspace_malloc(memptr->shm_mspace, bytes)) != NULL)
                break;
            else
                ERROR("EPLIB malloc failed to allocate %ld bytes\n", bytes);
        }
        i++;
    }

    DEBUG_ASSERT(mem);
    MAKE_BOUNDS(mem, bytes);

    return mem;
}

void* memory_realloc(void* ptr, size_t new_size)
{
    void* mem = NULL;
    int i = 0;

    if (!use_allocator)
    {
        memory_check_limits(new_size, 0);
        return realloc(ptr, new_size);
    }

    while (mem == NULL)
    {
        memory_t* memptr = &memory_table[i];
        /* First try to allocate in one of the existing chunks */
        if (memptr->in_use == 1)
            if ((mem = mspace_realloc(memptr->shm_mspace, ptr, new_size)) != NULL)
                break;

        /* Create a new chunck if already past the last valid chunk */
        if (i > memory_table_last)
        {
            memptr = memory_expand(new_size);
            if ((mem = mspace_realloc(memptr->shm_mspace, ptr, new_size)) != NULL)
                break;
            else
                ERROR("EPLIB realloc failed to allocate %ld bytes\n", new_size);
        }
        i++;
    }

    DEBUG_ASSERT(mem);
    MAKE_BOUNDS(mem, new_size);

    return mem;
}

void* memory_calloc(size_t num, size_t elem_size)
{
    void* mem = NULL;
    int i = 0;

    if (!use_allocator)
    {
        memory_check_limits(num * elem_size, 0);
        return calloc(num, elem_size);
    }

    size_t bytes = num * elem_size;
    while (mem == NULL)
    {
        memory_t* memptr = &memory_table[i];
        /* First try to allocate in one of the existing chunks */
        if (memptr->in_use == 1)
            if ((mem = mspace_calloc(memptr->shm_mspace, num, elem_size)) != NULL)
                break;

        /* Create a new chunck if already past the last valid chunk */
        if (i > memory_table_last)
        {
            memptr = memory_expand(bytes);
            if ((mem = mspace_calloc(memptr->shm_mspace, num, elem_size)) != NULL)
                break;
            else
                ERROR("EPLIB calloc failed to allocate %ld bytes\n", bytes);
        }
        i++;
    }

    DEBUG_ASSERT(mem);
    MAKE_BOUNDS(mem, bytes);

    return mem;
}

void* memory_memalign(size_t alignment, size_t bytes)
{
    void* mem = NULL;
    int i = 0;

    if (!use_allocator)
    {
        memory_check_limits(bytes, 0);
        MALLOC_ALIGN(mem, bytes, alignment);
        return mem;
    }

    while (mem == NULL)
    {
        memory_t* memptr = &memory_table[i];
        /* First try to allocate in one of the existing chunks */
        if (memptr->in_use == 1)
            if ((mem = mspace_memalign(memptr->shm_mspace, alignment, bytes)) != NULL)
                break;

        /* Create a new chunck if already past the last valid chunk */
        if (i > memory_table_last)
        {
            memptr = memory_expand(bytes);
            if ((mem = mspace_memalign(memptr->shm_mspace, alignment, bytes)) != NULL)
                break;
            else
                ERROR("EPLIB memalign failed to allocate %ld bytes\n", bytes);
        }
        i++;
    }

    DEBUG_ASSERT(mem);
    MAKE_BOUNDS(mem, bytes);

    return mem;
}

void memory_free(void* mem)
{
    if (!memory_table) return;

    if (!use_allocator)
    {
        FREE_ALIGN(mem);
        return;
    }

    int mem_idx;
    if (memory_is_shmem(mem, &mem_idx))
    {
        ASSERT((mem_idx >= 0 && mem_idx < memory_table_size));
        memory_t* memptr = &memory_table[mem_idx];
        if (memptr->in_use == 1)
        {
            REMOVE_BOUNDS(mem);
            mspace_free(memptr->shm_mspace, mem);
            return;
        }
    }

    ERROR("Memory address not allocated / registered with EPLIB %p, use __libc_free\n", mem);
}

#endif /* ENABLE_CLIENT_ONLY */

void memory_release(int memid)
{
    if (!use_allocator) return;

    memory_t* memptr = &memory_table[memid];
    if (memptr->in_use == 1)
    {
        if (memptr->shm_mspace != NULL)
            destroy_mspace(memptr->shm_mspace);
        if (memptr->shm_base != NULL)
            munmap(memptr->shm_base, memptr->mem_size);
        if (memptr->shm_id >= 0)
            close(memptr->shm_id);
        if (memptr->do_create == 1 && need_unlink)
            shm_unlink(memptr->shm_filename);

        memptr->in_use = 0;
        memptr->shm_base = NULL;
        memptr->client_shm_base = NULL;
        memptr->shm_mspace = NULL;
        memptr->shm_id = -1;
        memptr->mem_size = 0;
    }
}

void memory_unlink()
{
    if (!use_allocator) return;

    if (memory_table != NULL)
    {
        for (int memid = 0; memid < memory_table_size; memid++)
        {
            memory_t* memptr = &memory_table[memid];
            if (memptr->in_use) shm_unlink(memptr->shm_filename);
        }
    }
    need_unlink = 0;
}

void memory_finalize()
{
    if (!use_allocator) return;

    if (memory_table != NULL)
    {
        for (int memid = 0; memid < memory_table_size; memid++)
            memory_release(memid);
        FREE_ALIGN(memory_table);
        memory_table = NULL;
    }
    use_allocator = 0;
}
