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
#ifndef POINTER_CHECKER_HPP
#define POINTER_CHECKER_HPP

#include <list>

using namespace std;

namespace MLSL
{
    enum PointerCheckerResultType
    {
        PCRT_NONE         = 0,
        PCRT_UNKNOWN_PTR  = 1,
        PCRT_OUT_OF_RANGE = 2,
    };

    typedef struct BufInfo
    {
        char*  start;
        char*  stop;
        size_t size;
    } BufInfo;

    class PointerChecker
    {
    private:
        list<BufInfo> bufList;

    public:
        PointerChecker();
        ~PointerChecker();
        void Add(void* ptr, size_t size);
        void Remove(void* ptr);
        PointerCheckerResultType CheckInternal(void* ptr, size_t size);
        void Check(void* ptr, size_t size);
        void Print(PointerCheckerResultType result);
    };
}

#endif /* POINTER_CHECKER_HPP */
