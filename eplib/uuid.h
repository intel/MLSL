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
#ifndef UUID_H
#define UUID_H

#define UUID_DEFAULT "00FF00FF-0000-0000-0000-00FF00FF00FF"
#define UUID_STR_LEN sizeof(UUID_DEFAULT) // 37 = 36 (uuid) + 1 (\0)

char* get_uuid();

#endif /* UUID_H */
