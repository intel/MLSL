# Intel(R) Machine Learning Scaling Library for Linux* OS
[![Apache License, Version 2.0](https://img.shields.io/badge/license-Apache%20License,%20Version%202.0-green.svg)](LICENSE)
![v2018.0 Preview](https://img.shields.io/badge/v.2018.1-Preview-orange.svg)
## Introduction ##
Intel(R) Machine Learning Scaling Library (Intel(R) MLSL) is a library providing
an efficient implementation of communication patterns used in deep learning.

    - Built on top of MPI, allows for use of other communication libraries
    - Optimized to drive scalability of communication patterns
    - Works across various interconnects: Intel(R) Omni-Path Architecture,
      InfiniBand*, and Ethernet
    - Common API to support Deep Learning frameworks (Caffe*, Theano*,
      Torch*, etc.)

Intel(R) MLSL package comprises the Intel MLSL Software Development Kit (SDK)
and the Intel(R) MPI Library Runtime components.
## SOFTWARE SYSTEM REQUIREMENTS ##
This section describes the required software.

Operating Systems:

    - Red Hat* Enterprise Linux* 6 or 7
    - SuSE* Linux* Enterprise Server 12
    - Ubuntu* 16

Compilers:

    - GNU*: C, C++ 4.4.0 or higher
    - Intel(R) C++ Compiler for Linux* OS 16.0 through 17.0 or higher

Virtual Environments:
    - Docker*
    - KVM*
## Installing Intel(R) Machine Learning Scaling Library ##
Installing the Intel(R) MLSL using RPM Package Manager (root mode):

    1. Log in as root

    2. Install the package:

        $ rpm -i intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64.rpm

        where <version>.<update>-<package#> is a string, such as: 2017.0-009

    3. Uninstalling the Intel(R) MLSL using the RPM Package Manager

        $ rpm -e intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64

Installing the Intel(R) MLSL using install.sh (user mode):

    1. Run install.sh and follow the instructions.

    There is no uninstall script. To uninstall the Intel(R) MLSL, delete the
    full directory you have installed the package into.
Installing the Intel(R) MLSL using Makefile:

    1. make

    2. [MLSL_INSTALL_PATH=/path] make install
    By default MLSL_INSTALL_PATH=$PWD/_install

## Launching Sample Application ##

Before you start using the Intel(R) MLSL, make sure to set up the library environment.
Use the command:

    $ source <install_dir>/intel64/bin/mlslvars.sh
    $ cd <install_dir>/test
    $ make run

Log file with output will be in the same directory.
Here  <install_dir> is the Intel MLSL installation directory.

## License ##
Intel MLSL is licensed under [Intel Simplified Software License](https://github.com/01org/MLSL/blob/master/LICENSE).
## Optimization Notice ##
Intel's compilers may or may not optimize to the same degree for non-Intel
microprocessors for optimizations that are not unique to Intel microprocessors.
These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other
optimizations. Intel does not guarantee the availability, functionality, or
effectiveness of any optimization on microprocessors not manufactured by Intel.
Microprocessor-dependent optimizations in this product are intended for use 
with Intel microprocessors. Certain optimizations not specific to Intel 
microarchitecture are reserved for Intel microprocessors. Please refer to the 
applicable product User and Reference Guides for more information regarding the
specific instruction sets covered by this notice.

Notice revision #20110804
