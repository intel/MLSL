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
 * MPI Endpoints Window Object Interface
 *
 */

#include "client.h"
#include "common.h"
#include "debug.h"
#include "memory.h"
#include "window.h"

#define EP_WINDOW_TABLE_INC 1024
#define EP_WINDOW_EXCLUSIVE 1

window_t* window_table;
int window_table_size = 0;
pthread_mutex_t win_mutex;

#ifdef ENABLE_MPIRMA_ENDPOINTS

static void window_table_grow(int inc)
{
    int i;

    if (window_table == NULL)
        MALLOC_ALIGN(window_table, (window_table_size+inc) * sizeof(window_t), CACHELINE_SIZE);
    else
    {
        DEBUG_PRINT("WARNING: realloc loses cache alignment\n");
        window_table = (window_t*)realloc(window_table, (window_table_size+inc) * sizeof(window_t));
    }

    ASSERT(window_table != NULL);

    for (i = window_table_size; i < window_table_size + inc; i++)
       window_table[i].in_use = 0;

    window_table_size += inc;
}

void window_init()
{
    pthread_mutex_init(&win_mutex, NULL);
    window_table_grow(EP_WINDOW_TABLE_INC);
}

void window_register(MPI_Win* win_ptr, void* base_ptr, client_t* myclient, int free_at_release)
{
    int i;

    int mtx_ret = pthread_mutex_lock(&win_mutex);
    assert(mtx_ret == 0);

    for (i = 0; i < window_table_size; i++)
       if (!window_table[i].in_use) break;

    if (i == window_table_size)
       window_table_grow(EP_WINDOW_TABLE_INC);

    ASSERT(i != *win_ptr);
    window_table[i].in_use = 1;
    window_table[i].win = *win_ptr;
    window_table[i].base = *(void**)base_ptr;
    window_table[i].client = myclient;
    window_table[i].free_at_release = free_at_release;
    if (myclient != NULL)
       *win_ptr = i;

    mtx_ret = pthread_mutex_unlock(&win_mutex);
    assert(mtx_ret == 0);
}

client_t* window_get_client(MPI_Win win)
{
#ifdef EP_WINDOW_EXCLUSIVE
    DEBUG_ASSERT(window_table[(int)win].win != win);
    DEBUG_ASSERT(window_table[(int)win].client != NULL);
    return window_table[(int)win].client;
#else
    int i;

    int mtx_ret = pthread_mutex_lock(&win_mutex);
    assert(mtx_ret == 0);

    for (i = 0; i < window_table_size; i++)
    {
        if (window_table[i].in_use)
        {
            if (i == win && window_table[i].win != win)
            {
                DEBUG_ASSERT(window_table[i].client != NULL);
                mtx_ret = pthread_mutex_unlock(&win_mutex);
                assert(mtx_ret == 0);
                return window_table[i].client;
            }
            else if (window_table[i].win == win)
            {
                DEBUG_ASSERT(window_table[i].client == NULL);
                mtx_ret = pthread_mutex_unlock(&win_mutex);
                assert(mtx_ret == 0);
                return NULL;
            }
        }
    }

    ERROR("Invalid Window Object\n");
    return NULL;
#endif
}

MPI_Win window_get_server_win(MPI_Win win)
{
    DEBUG_ASSERT(window_table[(int)win].in_use);
    return window_table[(int)win].win;
}

void window_release(int win)
{
    int i;

    int mtx_ret = pthread_mutex_lock(&win_mutex);
    assert(mtx_ret == 0);

    for (i = 0; i < window_table_size; i++)
    {
        if (window_table[i].in_use)
        {
            if (i == win && window_table[i].win != win)
            {
                ASSERT(window_table[i].client != NULL);
                window_table[i].in_use = 0;
                if (window_table[i].free_at_release) memory_free(window_table[i].base);
                break;
            }
            else if (window_table[i].win == win)
            {
                ASSERT(window_table[i].client == NULL);
                window_table[i].in_use = 0;
                break;
            }
        }
    }

    mtx_ret = pthread_mutex_unlock(&win_mutex);
    assert(mtx_ret == 0);
}

void window_finalize()
{
    if (window_table != NULL)
    {
        if (window_table_size == EP_WINDOW_TABLE_INC)
            FREE_ALIGN(window_table);
        else
            free(window_table);
    }
}

#else

void window_init() {}
void window_finalize() {}

#endif /* ENABLE_MPIRMA_ENDPOINTS */
