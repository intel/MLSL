# Intel(R) Machine Learning Scaling Library for Linux* OS
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
Installing Intel(R) MLSL by building from source:

        $ make all
        $ [MLSL_INSTALL_PATH=/path] make install

    By default MLSL_INSTALL_PATH=$PWD/_install

Binary releases are available on our [release page](https://github.com/intel/MLSL/releases).

Installing Intel(R) MLSL using RPM Package Manager (root mode):

    1. Log in as root

    2. Install the package:

        $ rpm -i intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64.rpm

        where <version>.<update>-<package#> is a string, such as: 2017.0-009

    3. Uninstalling Intel(R) MLSL using the RPM Package Manager

        $ rpm -e intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64

Installing Intel(R) MLSL using the tar file (user mode):

        $ tar zxf l_mlsl-devel-64-<version>.<update>.<package#>.tgz
        $ cd l_mlsl_<version>.<update>.<package#>
        $ ./install.sh

    There is no uninstall script. To uninstall Intel(R) MLSL, delete the
    full directory you have installed the package into.

## Launching Sample Application ##

The sample application needs python with the numpy package installed.
You can use [Intel Distribution for Python]
(https://software.intel.com/en-us/distribution-for-python),
[Anaconda](https://conda.io/docs/user-guide/install/download.html),
or the python and numpy that comes with your OS.
Before you start using Intel(R) MLSL, make sure to set up the library environment.

Use the command:

    $ source <install_dir>/intel64/bin/mlslvars.sh
    $ cd <install_dir>/test
    $ make run

If the test fails, look in the log files in the same directory.
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

*Other names and brands may be claimed as the property of others.
