-------------------------------------------------------
Intel(R) Machine Learning Scaling Library for Linux* OS
README
-------------------------------------------------------

------------
Introduction
------------

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

----------------------------------------------------
Installing Intel(R) Machine Learning Scaling Library
----------------------------------------------------

I.   Installing Intel(R) MLSL using RPM Package Manager (root mode):

        1. Log in as root.

        2. Install the package:
            $ rpm -i intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64.rpm

            where <version>.<update>-<package#> is a string, such as: 2017.0-009

     To uninstall Intel(R) MLSL, use the command:
     $ rpm -e intel-mlsl-devel-64-<version>.<update>-<package#>.x86_64

II.  Installing Intel(R) MLSL using install.sh (user mode):

     Run install.sh and follow the instructions.

     There is no uninstall script. To uninstall Intel(R) MLSL, delete the entire
     directory where you have installed the package.

-------------------
Directory Structure
-------------------

Following a successful installation, the files associated with the Intel(R) MLSL
are installed on your host system. The following directory map indicates the 
default structure and identifies the file types stored in each sub-directory:

`-- opt
    `-- intel        Common directory for Intel(R) Software Development Products.
            `-- mlsl_<version>.<update>.<package#>
                |               Subdirectory for the version, specific update
                |               and package number of Intel(R) MLSL.
                |-- doc         Subdirectory with documentation.
                |-- |-- API_Reference.htm
                |   |-- Developer_Guide.pdf
                |   |-- README.txt
                |   |-- Release_Notes.txt
                |   |-- api     Subdirectory with API reference.
                |-- example     Intel(R) MLSL example
                |   |-- Makefile
                |   `-- mlsl_example.cpp
                |-- intel64     Files for specific architecture.
                |   |-- bin     Binaries, scripts, and executable files.
                |   |   |-- ep_server
                |   |   |-- mlslvars.sh
                |   |   |-- process
                |   |   |   |-- hydra_persist
                |   |   |   |-- mpiexec -> mpiexec.hydra
                |   |   |   |-- mpiexec.hydra
                |   |   |   |-- mpirun
                |   |   |   `-- pmi_proxy
                |   |   `-- thread
                |   |       |-- fi_info
                |   |       |-- hydra_bstrap_proxy
                |   |       |-- hydra_nameserver
                |   |       |-- hydra_pmi_proxy
                |   |       |-- mpiexec -> mpiexec.hydra
                |   |       |-- mpiexec.hydra
                |   |       `-- mpirun
                |   |-- etc      Configuration files.
                |   |   |-- mpiexec.conf
                |   |   |-- tmi.conf
                |   |   |-- tuning_knl_ofi.dat
                |   |   |-- tuning_knl_shm-ofi.dat
                |   |   |-- tuning_skx_ofi.dat
                |   |   `-- tuning_skx_shm-ofi.dat
                |   |-- include   Include and header files.
                |   |   |-- mlsl  Subdirectory for Python* module
                |   |   |   |-- __init__.py
                |   |   |   `-- mlsl.py
                |   |   |-- mlsl.h
                |   |   `-- mlsl.hpp
                |   `-- lib 
                |       |-- libmlsl.so -> process/libmlsl.so.1.0
                |       |-- libmlsl.so.1 -> process/libmlsl.so.1.0
                |       |-- process
                |       |   |-- libmlsl.so -> libmlsl.so.1
                |       |   |-- libmlsl.so.1 -> libmlsl.so.1.0
                |       |   |-- libmlsl.so.1.0
                |       |   |-- libmpi.so -> libmpi.so.12.0
                |       |   |-- libmpi.so.12 -> libmpi.so.12.0
                |       |   |-- libmpi.so.12.0
                |       |   |-- libtmip_psm2.so -> libtmip_psm2.so.1.0
                |       |   |-- libtmip_psm2.so.1.0
                |       |   |-- libtmip_psm.so -> libtmip_psm.so.1.2
                |       |   |-- libtmip_psm.so.1.0 -> libtmip_psm.so.1.2
                |       |   |-- libtmip_psm.so.1.1 -> libtmip_psm.so.1.2
                |       |   |-- libtmip_psm.so.1.2
                |       |   |-- libtmi.so -> libtmi.so.1.2
                |       |   |-- libtmi.so.1.0 -> libtmi.so.1.2
                |       |   |-- libtmi.so.1.1 -> libtmi.so.1.2
                |       |   `-- libtmi.so.1.2
                |       `-- thread
                |           |-- libfabric.so -> libfabric.so.1
                |           |-- libfabric.so.1
                |           |-- libmlsl.so -> libmlsl.so.1
                |           |-- libmlsl.so.1 -> libmlsl.so.1.0
                |           |-- libmlsl.so.1.0
                |           |-- libmpi.so -> libmpi.so.12.0.0
                |           |-- libmpi.so.12 -> libmpi.so.12.0.0
                |           |-- libmpi.so.12.0 -> libmpi.so.12.0.0
                |           |-- libmpi.so.12.0.0
                |           `-- prov
                |               |-- libpsmx2-fi.so
                |               |-- librxm-fi.so
                |               |-- libsockets-fi.so
                |               `-- libverbs-fi.so
                |-- licensing
                |   |-- mlsl    Subdirectory for supported files, license
                |   |           of the Intel(R) MLSL
                |   |   |-- LICENSE
                |   |   `-- third-party-programs.txt
                |   `-- mpi     Subdirectory for supported files, EULAs,
                |               redist files, third-party-programs file 
                |               of the Intel(R) MPI Library
                |       |-- process
                |       |   |-- license.txt
                |       |   `-- third-party-programs.txt
                |       `-- thread
                |           |-- license.txt
                |           `-- third-party-programs.txt
                `-- test       Intel(R) MLSL tests
                    |-- cmlsl_test.c
                    |-- Makefile
                    |-- mlsl_test.cpp
                    `-- mlsl_test.py

--------------------------------
Disclaimer and Legal Information
--------------------------------
No license (express or implied, by estoppel or otherwise) to any intellectual
property rights is granted by this document.

Intel disclaims all express and implied warranties, including without limitation,
the implied warranties of merchantability, fitness for a particular purpose, and
non-infringement, as well as any warranty arising from course of performance,
course of dealing, or usage in trade.

This document contains information on products, services and/or processes in
development. All information provided here is subject to change without notice.
Contact your Intel representative to obtain the latest forecast, schedule,
specifications and roadmaps.

The products and services described may contain defects or errors known as
errata which may cause deviations from published specifications. Current
characterized errata are available on request.

No computer software can provide absolute security. End users are responsible for
securing their own deployment of computer software in any environment.

Intel, Intel Core, Xeon, Xeon Phi and the Intel logo are trademarks of Intel
Corporation in the U.S. and/or other countries.

* Other names and brands may be claimed as the property of others.

(C) Intel Corporation.

Optimization Notice
-------------------

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
