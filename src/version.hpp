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
#ifndef VERSION_HPP
#define VERSION_HPP

#define STR_HELPER(x) #x
#define STR(x)        STR_HELPER(x)

#define MLSL_PACKAGE_VERSION_PREFIX  "Intel(R) Machine Learning Scaling Library for Linux* OS, Version "
#define MLSL_PACKAGE_VERSION_POSTFIX "Copyright (c) " STR(MLSL_YEAR) ", Intel Corporation. All rights reserved."
#define MLSL_PACKAGE_VERSION         MLSL_PACKAGE_VERSION_PREFIX       \
                                     STR(MLSL_OFFICIAL_VERSION)        \
                                     " (" STR(MLSL_FULL_VERSION) ")\n" \
                                     MLSL_PACKAGE_VERSION_POSTFIX

#endif /* VERSION_HPP */
