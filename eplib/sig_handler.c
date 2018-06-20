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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "sig_handler.h"

#ifdef ENABLE_CLIENT_ONLY
#include "eplib.h"
#endif

static struct sigaction SIGSEGV_old_act;
static struct sigaction SIGBUS_old_act;
static struct sigaction SIGILL_old_act;
static struct sigaction SIGABRT_old_act;
static struct sigaction SIGINT_old_act;
static struct sigaction SIGTERM_old_act;
static struct sigaction act;

#ifdef ENABLE_CLIENT_ONLY
void atexit_handler(void)
{
    EPLIB_finalize();

    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);

    if (is_mpi_inited && !is_mpi_finalized)
        PMPI_Finalize();
}
#endif

void sig_handler(int sig_num)
{
#ifdef ENABLE_CLIENT_ONLY
    atexit_handler();
    _exit(1);
#else
    /* Skip signals on server side, server will be stoped over command queue by client */
#endif
}

void init_sig_handlers(void)
{
    act.sa_handler = sig_handler;

    (void)sigaction(SIGSEGV, &act, &SIGSEGV_old_act);
    (void)sigaction(SIGBUS,  &act, &SIGBUS_old_act);
    (void)sigaction(SIGILL,  &act, &SIGILL_old_act);
    (void)sigaction(SIGABRT, &act, &SIGABRT_old_act);
    (void)sigaction(SIGINT,  &act, &SIGINT_old_act);
    (void)sigaction(SIGTERM, &act, &SIGTERM_old_act);
}

void fini_sig_handlers(void)
{
    (void)sigaction(SIGSEGV, &SIGSEGV_old_act, NULL);
    (void)sigaction(SIGBUS,  &SIGBUS_old_act, NULL);
    (void)sigaction(SIGILL,  &SIGILL_old_act, NULL);
    (void)sigaction(SIGABRT, &SIGABRT_old_act, NULL);
    (void)sigaction(SIGINT,  &SIGINT_old_act, NULL);
    (void)sigaction(SIGTERM, &SIGTERM_old_act, NULL);
}
