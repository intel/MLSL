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
#include "log.hpp"
#include "mlsl.hpp"
#include "pointer_checker.hpp"

namespace MLSL
{
    PointerChecker::PointerChecker() {}

    PointerChecker::~PointerChecker()
    {
        MLSL_ASSERT(bufList.empty(), "all buffers should be freed explicitly");
    }

    void PointerChecker::Add(void* ptr, size_t size)
    {
        MLSL_LOG(DEBUG, "ptr %p, size %zu", ptr, size);
        MLSL_ASSERT(CheckInternal(ptr, size) == PCRT_UNKNOWN_PTR, "this pointer already allocated (ptr %p, sz %zu)", ptr, size);

        BufInfo bufInfo;
        bufInfo.start = (char*)ptr;
        bufInfo.stop = (char*)ptr + size;
        bufInfo.size = size;
        bufList.push_back(bufInfo);
    }

    void PointerChecker::Remove(void* ptr)
    {
        MLSL_LOG(DEBUG, "ptr %p", ptr);
        MLSL_ASSERT(!bufList.empty(), "attempt to remove pointer (%p) from empty list", ptr);

        list<BufInfo>::iterator it = bufList.begin();
        for (; it != bufList.end(); it++)
        {
            if ((*it).start == (char*)ptr) break;
        }
        MLSL_ASSERT(it != bufList.end(), "unknown pointer");
        bufList.erase(it);
    }

    PointerCheckerResultType PointerChecker::CheckInternal(void* ptr, size_t size)
    {
        MLSL_LOG(DEBUG, "ptr %p, size %zu", ptr, size);

        PointerCheckerResultType res = PCRT_NONE;

        list<BufInfo>::iterator it = bufList.begin();
        for(; it != bufList.end(); it++)
        {
            if ((*it).start <= (char*)ptr && (*it).stop >= (char*)ptr)
            {
                if ((*it).stop >= (char*)ptr + size)
                    res = PCRT_NONE;
                else
                {
                    size_t beyondBoundary = (size_t)(((char*)ptr + size) - (*it).stop);
                    MLSL_LOG(INFO, "ptr %p is %zu bytes beyond the array boundary %p", ((char*)ptr + size), beyondBoundary, (*it).stop);
                    res = PCRT_OUT_OF_RANGE;
                }
                break;
            }
        }

        if (it == bufList.end())
            res = PCRT_UNKNOWN_PTR;

        return res;
    }

    void PointerChecker::Check(void* ptr, size_t size)
    {
        MLSL_LOG(DEBUG, "ptr %p, size %zu", ptr, size);
        Print(CheckInternal(ptr, size));
    }

    void PointerChecker::Print(PointerCheckerResultType result)
    {
        switch (result)
        {
            case PCRT_NONE:
                break;
            case PCRT_UNKNOWN_PTR:
                MLSL_LOG(DEBUG, "unknown ptr");
                break;
            case PCRT_OUT_OF_RANGE:
                MLSL_ASSERT(0, "out of range");
                break;
            default:
                MLSL_ASSERT(0, "unknown result (%d)", result);
        }
    }
}
