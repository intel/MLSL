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
#ifndef _WINDOW_H_
#define _WINDOW_H_

#include "client.h"

struct __window_t
{
    int in_use;
    MPI_Win win;
    void* base;
    client_t* client;        /* Pointer to client */
    int free_at_release;
    int pad[9];
};

typedef struct __window_t window_t;

void window_init();
void window_register(MPI_Win*, void*, client_t*, int);
client_t* window_get_client(MPI_Win);
void* window_get_baseptr(MPI_Win);
MPI_Win window_get_server_win(MPI_Win);
void window_release(int);
void window_finalize();

#endif /* _WINDOW_H_ */
