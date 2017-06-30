# Intel(R) Machine Learning Scaling Library for Linux* OS
[![Intel Simplified Sowtware License](https://img.shields.io/badge/license-Intel%20Simplified%20Software%20License-green.svg)](LICENSE)
![v2017.1 Preview](https://img.shields.io/badge/v.2017.1-Preview-orange.svg)
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
To install the package (root mode):

    1.  Log in as root.
    2.  Install the package:
          $ rpm --import PUBLIC_KEY.PUB
          $ rpm -i intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64.rpm
          where <version>.<update>-<package#> is a string, such as: 2017.0-009
    3.  Uninstalling the Intel(R) MLSL using the RPM Package Manager
          $ rpm -e intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64

To install the package (user mode):

    1.  Run install.sh and follow the instructions.

    There is no uninstall script. To uninstall the Intel(R) MLSL, delete the full
    directory you have installed the package into.
## License ##
Intel MLSL is licensed under 
[Intel Simplified Software License](https://github.com/01org/MLSL/blob/master/LICENSE).

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
