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
#ifndef TUNE_H
#define TUNE_H

void tune()
{
    /* PSM2 */
    setenv("HFI_NO_CPUAFFINITY",        "1",       0);
    setenv("PSM2_SHAREDCONTEXTS",       "0",       0);
    setenv("PSM2_MEMORY",               "large",   0);
    setenv("PSM2_TID_SENDSESSIONS_MAX", "2097152", 0);
    setenv("PSM2_RCVTHREAD",            "0",       0);
    setenv("PSM2_DEVICES",              "shm,hfi", 0);

    /* OFI/PSM2 */
    setenv("FI_PSM2_NAME_SERVER",        "0",      0);
    setenv("FI_PSM2_PROG_THREAD",        "0",      0);

    setenv("I_MPI_COLL_INTRANODE", "pt2pt", 0);
}

#endif /* TUNE_H */
