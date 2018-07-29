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
/** @file mlsl.hpp
 *  @brief Public external interface to Intel® Machine Learning Scaling Library (Intel® %MLSL)
 */

#ifndef MLSL_HPP
#define MLSL_HPP

#include <cstddef>

/* API version (which is not necessarily the same as the tarball/mlsl package version number) */

/** Intel %MLSL API major version. */
#define MLSL_MAJOR_VERSION 1

/** Intel %MLSL API minor version. */
#define MLSL_MINOR_VERSION 0

/**
 *  A macro to create a full API version from the major and minor API versions.
 *  @param major major API version
 *  @param minor minor API version
 */
#define MLSL_VERSION(major, minor) ((major << 16) | (minor))

/**
 *  A macro to retrieve the major API version from the full API version.
 *  @param version full API version
 */
#define MLSL_MAJOR(version)        (version >> 16)

/**
 *  A macro to retrieve the minor version from the full version.
 *  @param version full API version
 */
#define MLSL_MINOR(version)        (version & 0xFFFF)

/**
 *  A macro to check the condition (v1 >= v2).
 *  @param v1 the first full API version to compare
 *  @param v2 the second full API version to compare
 */
#define MLSL_VERSION_GE(v1, v2)    ((MLSL_MAJOR(v1) > MLSL_MAJOR(v2)) ||                                      \
                                    (MLSL_MAJOR(v1) == MLSL_MAJOR(v2) && MLSL_MINOR(v1) == MLSL_MINOR(v2)) || \
                                    (MLSL_MAJOR(v1) == MLSL_MAJOR(v2) && MLSL_MINOR(v1) > MLSL_MINOR(v2)))

/**
 *  A macro to check the condition (v1 < v2).
 *  @param v1 the first full API version to compare
 *  @param v2 the second full API version to compare
 */
#define MLSL_VERSION_LT(v1, v2)    ((MLSL_MAJOR(v1) < MLSL_MAJOR(v2)) ||                                   \
                                    (MLSL_MAJOR(v1) == MLSL_MAJOR(v2) && MLSL_MINOR(v1) < MLSL_MINOR(v2)))

/** A macro to prevent explicit creation of a class instance by user. */
#define NO_EXPLICIT_CREATION(ClassName)     \
  protected:                                \
    ClassName() {}                          \
    ~ClassName() {}                         \
  private:                                  \
    ClassName(const ClassName&);            \
    ClassName& operator=(const ClassName&); \

/**
 * @namespace MLSL
 * @brief The namespace containing all Intel %MLSL classes and types.
 */
namespace MLSL
{
    /** Type definition for a communication request type. */
    typedef int CommReq;

    /** The data type used for representing input/output activations and parameters (weights or biases). */
    enum DataType
    {
        DT_FLOAT  = 0,
        DT_DOUBLE = 1,
        DT_BYTE   = 2
    };

    /** Phases supported by Intel %MLSL. */
    enum PhaseType
    {
        PT_TRAIN = 0, /**< training phase */
        PT_TEST  = 1  /**< testing phase  */
    };

    /**
     *  @brief Groups of processes supported by Intel %MLSL
     *
     *  A group represents a subset of processes handling the model.<br>
     *  The data group contains processes working on the same part of parameters
     *  for different input batches (i.e. doing data parallelism)
     *  and communicating to exchange gradients with respect to parameters.<br>
     *  The model group contains processes working on different parts of parameters
     *  for the same input batch (i.e. doing model parallelism)
     *  and communicating to exchange activations and gradients with respect to activations.<br>
     *  The global group contains all Intel %MLSL processes.
     */
    enum GroupType
    {
        GT_DATA   = 0, /**< data group */
        GT_MODEL  = 1, /**< model group */
        GT_GLOBAL = 2  /**< global group */
    };

    /** Reduction operations for Distribution::Reduce(), Distribution::AllReduce() and Distribution::ReduceScatter(). */
    enum ReductionType
    {
        RT_SUM = 0,
        RT_MIN = 1,
        RT_MAX = 2
    };

    /**
     *  @brief Compute operation types
     *
     *  Each operation type defines specific relationship between the input and output activations
     *  and associated parameters (weights or biases).<br>
     *  (abbreviations: IA – input activation, OA – output activation)
     */
    enum OpType
    {
        OT_CC     = 0, /**< Cross-correlation - IA and OA independent and has parameters */
        OT_BIAS   = 1, /**< Bias - the same IA and OA (dependent) but has parameters */
        OT_ACT    = 2, /**< %Activation operation - Same IA and OA and no parameters */
        OT_POOL   = 3, /**< %Activation operation - Same IA and OA and no parameters */
        OT_SPLIT  = 4, /**< OA depends on IA (=OA1+OA2...) and no parameters */
        OT_CONCAT = 5, /**< OA depends on IA1+IA2+... and no parameters */
        OT_BCAST  = 6, /**< OA1=IA, OA2=IA, ... and no parameters */
        OT_REDUCE = 7, /**< OA=IA1+IA2+... and no parameters */
        OT_DATA   = 8, /**< Only OA, no IA */
        OT_EVAL   = 9  /**< Only IA, no OA */
    };

    /** Compression type. */
    enum CompressionType
    {
        CT_NONE         = 0, /**< Do not use compression*/
        CT_QUANTIZATION = 1  /**< Use quantization*/
    };

    /**
     *  @brief A struct to hold quantization parameters
     *
     *  Holds information about the quantization library and functions.
     */
    typedef struct
    {
        char* lib_path;                 /**< Quantization library path. */
        char* quant_buffer_func_name;   /**< Name of the function for buffer quantization. */
        char* dequant_buffer_func_name; /**< Name of the function for buffer dequantization. */
        char* reduce_sum_func_name;     /**< Name of the function for reduction of quantized buffers. */
        size_t block_size;              /**< Quantization meta data: block size in bytes. */
        size_t elem_in_block;           /**< Quantization meta data: number of elements in a block. */
    } QuantParams;

    /**
     *  @brief A class to hold block information for activations packing/unpacking
     *
     *  Holds information about Activation partitioning and is used for packing/upacking to/from the communication buffer.
     */
    class CommBlockInfo
    {
        NO_EXPLICIT_CREATION(CommBlockInfo)

    public:

        /** @returns The offset of the mini-batch portion. */
        size_t GetMbOffset();

        /** @returns The length of the mini-batch portion. */
        size_t GetMbCount();

        /** @returns The offset of the feature map portion. */
        size_t GetFmOffset();

        /** @returns The length of the feature map portion. */
        size_t GetFmCount();

        /** @returns The size of a feature map in MLSL::DataType elements. */
        size_t GetFmSize();

        /** @returns The datatype of the feature map elements. */
        DataType GetDataType();

        /** @returns The offset in MLSL::DataType elements within the communication buffer where to pack to/unpack from. */
        size_t GetBufOffset();
    };

    /**
     *  @brief A wrapper class for operation input and output activations
     *
     *  Holds information about the input/output activation shape and allows performing associated communications.
     */
    class Activation
    {
        NO_EXPLICIT_CREATION(Activation)

    public:

        /** @returns The global count of feature maps. */
        size_t GetGlobalFmCount();

        /** @returns The offset of the local portion of feature maps in the global count of feature maps. */
        size_t GetGlobalFmOffset();

        /** @returns The local count of feature maps (the length of the local portion being processed by this process). */
        size_t GetLocalFmCount();

        /** @returns The count of data blocks being sent for this Activation instance. */
        size_t GetPackBlockCount();

        /** @returns The count of data blocks being received for this Activation instance. */
        size_t GetUnpackBlockCount();

        /**
         *  Returns the CommBlockInfo object containing information for packing.
         *  @param idx the object index
         *  @returns Block information for packing.
         */
        CommBlockInfo* GetPackBlock(size_t idx);

        /**
         *  Returns the CommBlockInfo object containing information for unpacking.
         *  @param idx the object index
         *  @returns Block information for unpacking.
         */
        CommBlockInfo* GetUnpackBlock(size_t idx);

        /** @returns The data type of the feature map elements. */
        DataType GetDataType();

        /** @returns The size of a feature map in MLSL::DataType elements. */
        size_t GetFmSize();

        /** @returns A pointer to internally allocated buffer sufficient for packing/unpacking. */
        void* GetCommBuf();

        /** @returns The size of internally allocated buffer sufficient for packing/unpacking. */
        size_t GetCommBufSize();

        /**
         *  Starts the non-blocking activation/gradient exchange with respect to activation.
         *  @param buf the buffer containing the packed activation
         */
        void StartComm(void* buf);

        /**
         *  Waits for completion of the activation/gradient exchange with respect to activation.
         *  @returns A pointer to the buffer containing the activation to be unpacked.
         */
        void* WaitComm();
    };

    /**
     *  @brief A wrapper class for operation parameters
     *
     *  Holds information about the shape of learnable parameters and allows performing associated communications.<br>
     *  Weights and biases should have separate instances.
     */
    class ParameterSet
    {
        NO_EXPLICIT_CREATION(ParameterSet)

    public:

        /** @returns The global count of kernels. */
        size_t GetGlobalKernelCount();

        /** @returns The offset of the local portion of kernels in the global count of kernels. */
        size_t GetGlobalKernelOffset();

        /** @returns The local count of kernels (the length of the local portion being processed by this process). */
        size_t GetLocalKernelCount();

        /**
         *  @returns The count of kernels on which this process performs synchronous Stochastic Gradient Descent (SGD).<br>
         *  Differs from local kernel count only when distributedUpdate = true.
         */
        size_t GetOwnedKernelCount();

        /** @returns The offset of the owned portion of kernels in the local count of kernels. */
        size_t GetOwnedKernelOffset();

        /** @returns The data type of the kernel elements. */
        DataType GetDataType();

        /** @returns The size of a kernel in DataType elements. */
        size_t GetKernelSize();

        /**
         *  @returns True if the exchange for the current parameter set is split into 2 communications (ReduceScatter for gradients + AllGather for increments)
         *  instead of 1 communication (AllReduce for gradients), false otherwise.
         */
        bool IsDistributedUpdate();

        /**
         *  Starts the non-blocking exchange of the gradient with respect to parameters.
         *  @param buf the buffer containing the gradient
         */
        void StartGradientComm(void* buf);

        /**
         *  Starts the non-blocking exchange of parameters increment. Applicable only when distributedUpdate = true.
         *  @param buf the buffer containing the increment
         */
        void StartIncrementComm(void* buf);

        /** Waits for completion of the exchange of gradients with respect to parameters.
         *  @returns A pointer to the buffer containing the aggregated gradients with respect to parameters.
         */
        void* WaitGradientComm();

        /** Tests for completion of the exchange of gradients with respect to parameters.
         *  @param isCompleted the completion status of the request, true if request is completed, false otherwise
         *  @returns A pointer to the buffer containing the aggregated gradients with respect to parameters if request is completed,
         *           NULL otherwise
         */
        void* TestGradientComm(bool* isCompleted);

        /** Waits for completion of the exchange of parameters increment.
         *  @returns A pointer to the buffer containing the increment obtained with the synchronous SGD.
         */
        void* WaitIncrementComm();
    };

    /**
     *  @brief A class to hold information about the parallelism scheme being used
     *
     *  Holds information about the parallelism scheme currently in use (data/model/hybrid parallelism)
     *  and provides the low-level API for collectives over data/model/global groups of processes.<br>
     *  These collectives can be used, for example, to broadcast initial parameters, and to collect the
     *  snapshot in model/hybrid parallelism.
     */
    class Distribution
    {
        NO_EXPLICIT_CREATION(Distribution)

    public:

        /**
         *  Returns the process index within a group.
         *  @param groupType the group type
         *  @returns The process index.
         */
        size_t GetProcessIdx(GroupType groupType);

        /**
         *  Returns the number of processes within a group.
         *  @param groupType the group type
         *  @returns The number of processes.
         */
        size_t GetProcessCount(GroupType groupType);

        /**
         *  Broadcasts a values from the root process to all other processes of the group.
         *  @param buffer the starting address of the buffer
         *  @param count the number of elements in the buffer
         *  @param dataType the data type of the buffer elements
         *  @param rootIdx the index of the root process within a group
         *  @param groupType group type
         *  @returns A communication request.
         */
        CommReq* Bcast(void* buffer, size_t count, DataType dataType, size_t rootIdx, GroupType groupType);

        /**
         *  Reduces values on all processes within a group.
         *  @param sendBuffer the address of the send buffer
         *  @param recvBuffer the address of the receive buffer, meaningful only at the root process
         *  @param count the number of elements in send/receive buffers
         *  @param dataType data the type of buffers' elements
         *  @param redType the reduction operation type
         *  @param rootIdx the index of the root process within a group
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* Reduce(void* sendBuffer, void* recvBuffer, size_t count, DataType dataType, ReductionType redType, size_t rootIdx, GroupType groupType);

        /**
         *  Reduces values from all processes of the group and distributes the result back to all the processes.
         *  @param sendBuffer the address of the send buffer
         *  @param recvBuffer the address of the receive buffer, meaningful only at the root process
         *  @param count the number of elements in send/receive buffers
         *  @param dataType the data type of buffer's elements
         *  @param redType the reduction operation type
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* AllReduce(void* sendBuffer, void* recvBuffer, size_t count, DataType dataType, ReductionType redType, GroupType groupType);

        /**
         *  Sends values from all processes of the group to all processes of the group.
         *  @param sendBuffer the address of the send buffer
         *  @param sendCount the number of elements in the send group buffer,
         *                   the send buffer should be large enough to hold messages for all processes of the group (at least (@c sendCount * @c GetProcessCount(groupType)) elements)
         *  @param recvBuffer the address of the receive buffer,
         *                    the receive buffer should be large enough to hold messages from all processes of the group
         *  @param dataType the data type of buffers' elements
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* AlltoAll(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType groupType);

        /**
         *  Gathers data from and scatters data to all members of a group.
         *  @param sendBuffer the address of the send buffer
         *  @param sendCounts the number of data elements that the process sends in the buffer that is specified in the sendbuf parameter
         *  @param sendOffsets the location, relative to the sendbuf parameter, of the data for each communicator process
         *  @param recvBuffer the address of receive buffer
         *                    the receive buffer should be large enough to hold messages from all processes of the group
         *  @param recvCounts the number of data elements from each communicator process in the receive buffer
         *  @param recvOffsets the location, relative to the recvbuf parameter, of the data from each communicator process
         *  @param dataType the data type of buffers' elements
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* AlltoAllv(void* sendBuffer, size_t* sendCounts, size_t* sendOffsets, void* recvBuffer, size_t* recvCounts, size_t* recvOffsets, DataType dataType, GroupType groupType);

        /**
         *  Gathers values from all processes of the group to the root process.
         *  @param sendBuffer the address of the send buffer
         *  @param sendCount the number of elements in the send buffer
         *  @param recvBuffer the address of the receive buffer, meaningful only at the root process,
         *                    the receive buffer should be large enough to hold messages from all processes of the group
         *  @param dataType the data type of buffers' elements
         *  @param rootIdx the index of the root process within a group
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* Gather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, size_t rootIdx, GroupType groupType);

        /**
         *  Gathers values from all processes of the group and distribute the combined values to all processes of the group.
         *  @param sendBuffer the address of the send buffer
         *  @param sendCount the number of elements in the send buffer
         *  @param recvBuffer the address of the receive buffer,
         *                    the receive buffer should be large enough to hold messages from all processes of the group
         *  @param dataType the data type of buffers' elements
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* AllGather(void* sendBuffer, size_t sendCount, void* recvBuffer, DataType dataType, GroupType groupType);

        /**
        *  Gathers values from all processes of the group and distribute the combined values to all processes of the group.
        *  @param sendBuffer the address of the send buffer
        *  @param sendCount the number of elements in the send buffer
        *  @param recvBuffer the address of the receive buffer,
        *                    the receive buffer should be large enough to hold messages from all processes of the group
        *  @param recvCounts the number of data elements from each communicator process in the receive buffer
        *  @param dataType the data type of buffers' elements
        *  @param groupType the group type
        *  @returns A communication request.
        */
        CommReq* AllGatherv(void* sendBuffer, size_t sendCount, void* recvBuffer, size_t* recvCounts, DataType dataType, GroupType groupType);

        /**
         *  Sends values from the root process to all processes of the group.
         *  @param sendBuffer the address of the send buffer, meaningful only at the root process,
         *                    the send buffer should be large enough to hold messages for all processes of the group
         *  @param recvBuffer the address of the receive buffer
         *  @param recvCount the number of elements in the receive buffer
         *  @param dataType the data type of buffers' elements
         *  @param rootIdx the index of the root process within a group
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* Scatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, size_t rootIdx, GroupType groupType);

        /**
         *  Combines values from all processes of the group and scatters the results in blocks back to all the processes.
         *  @param sendBuffer the address of the send buffer
         *  @param recvBuffer the address of the receive buffer
         *  @param recvCount the number of elements in the receive buffer (the number of elements in a block)
         *  @param dataType the data type of buffers' elements
         *  @param redType the reduction operation type
         *  @param groupType the group type
         *  @returns A communication request.
         */
        CommReq* ReduceScatter(void* sendBuffer, void* recvBuffer, size_t recvCount, DataType dataType, ReductionType redType, GroupType groupType);

        /**
         *  Sets a barrier for all the processes within a group.
         *  @param groupType the group type
         */
        void Barrier(GroupType groupType);
    };

    /**
     *  @brief A class to hold Operation registration information
     *
     *  Holds the information about learnable parameters and activation shapes and is used for creation of an Operation object.
     *  All the input/output activation shapes and parameter shapes (if any) should be added before calling Session::AddOperation()
     */
    class OperationRegInfo
    {
        NO_EXPLICIT_CREATION(OperationRegInfo)

    public:

        /**
         *  Sets the operation name (for debugging purposes)
         *  @param name the operation name
         */
        void SetName(const char* name);

        /**
         *  Adds an input activation shape to the operation.
         *  @param featureMapCount the number of feature maps
         *  @param featureMapSize the size of feature maps in MLSL::DataType elements
         *  @param dataType the data type of feature map elements
         *  @returns The index of the input activation.
         */
        size_t AddInput(size_t featureMapCount, size_t featureMapSize, DataType dataType);

        /**
         *  Adds an output activation shape to the operation.
         *  @param featureMapCount the number of feature maps
         *  @param featureMapSize the size of feature maps in MLSL::DataType elements
         *  @param dataType the data type of feature map's elements
         *  @returns The index of the output activation.
         */
        size_t AddOutput(size_t featureMapCount, size_t featureMapSize, DataType dataType);

        /**
         *  Adds a parameter set shape to the operation.
         *  @param kernelCount the number of kernels
         *  @param kernelSize the size of a single kernel in MLSL::DataType elements
         *  @param dataType the data type of kernel's elements
         *  @param distributedUpdate tells whether to use the distributed update of parameters (ReduceScatter() + AllGather() instead of AllReduce())
         *  @param compressType the type of compression (Environment::SetQuantizationParams() must be called before AddParameterSet() with quantization enabled)
         *  @returns The index of the parameter set.
         */
        size_t AddParameterSet(size_t kernelCount, size_t kernelSize, DataType dataType, bool distributedUpdate = false, CompressionType compressType = CT_NONE);

        /**
         *  Validates the activations/parameter sets information.
         *  @param dist the distribution (currently not used)
         */
        void Validate(Distribution* dist = NULL);
    };

    class Session;

    /**
     *  @brief A class to hold information about learnable parameters (parameter sets) and activations
     *         corresponding to a certain operation of the computational graph
     */
    class Operation
    {
        NO_EXPLICIT_CREATION(Operation)

    public:

        /** 
         *  Sets the operation's distribution for the case when the operation has been created
         *  without the distribution through @c session->AddOperation(regInfo, NULL)
         */
        void SetDistribution(Distribution* dist);

        /** @returns The distribution used by the current operation. */
        Distribution* GetDistribution();

        /** @returns The session to which the current operation belongs. */
        Session* GetSession();

        /** @returns The operation type. */
        OpType GetOpType();

        /**
         *  Sets the previous operation in the computational graph.
         *  @param prev the previous operation
         *  @param actIdx the index of the current operation's input activation correspoding to the previous operation
         *  @param prevOpActIdx the index of the previous operation's output activation correspoding to the current operation
         */
        void SetPrev(Operation* prev, size_t actIdx, size_t prevOpActIdx);

        /**
         *  Sets the next operation in the computational graph.
         *  @param next the next operation
         *  @param actIdx the index of the current operation's output activation correspoding to the next operation
         *  @param nextOpActIdx the index of the next operation's input activation correspoding to the current operation
         */
        void SetNext(Operation* next, size_t actIdx, size_t nextOpActIdx);

        /** @returns The operation name. */
        const char* GetName();

        /** @returns The length of the global mini-batch. */
        size_t GetGlobalMinibatchSize();

        /** @returns The length of the local mini-batch portion. */
        size_t GetLocalMinibatchSize();

        /** @returns The offset of the local mini-batch portion within the global mini-batch. */
        size_t GetGlobalMinibatchOffset();

        /** @returns The number of input activations for the current operation. */
        size_t GetInputCount();

        /**
         *  Returns the input activation by index.
         *  @param idx the input activation's index
         *  @returns The input activation.
         */
        Activation* GetInput(size_t idx);

        /** @returns The number of output activations for the current operation. */
        size_t GetOutputCount();

        /**
         *  Returns the output activation by index.
         *  @param idx the output activation's index
         *  @returns The output activation.
         */
        Activation* GetOutput(size_t idx);

        /** @returns True if the current operation has parameter sets (weights or bias), false otherwise. */
        bool HasParameterSets();

        /** @returns The number of parameter sets for the current operation. */
        size_t GetParameterSetCount();

        /**
         *  Returns the parameter set by index
         *  @param idx the parameter set's index
         *  @returns The parameter set.
         */
        ParameterSet* GetParameterSet(size_t idx);
    };

    /**
     *  @brief A class to measure and store performance statistics of communication
     *         among processes that perform computation in the computational graph
     */
    class Statistics
    {
        NO_EXPLICIT_CREATION(Statistics)

    public:

        /** Starts statistics collection. */
        void Start();

        /** Stops statistics collection. */
        void Stop();

        /** Clears the measured statistics information. */
        void Reset();

        /**  @returns True if the statistics collection is started, false otherwise. */
        bool IsStarted();

        /**  @returns True if statistics collection is enabled, false otherwise. */
        bool IsEnabled();

        /** Prints the measured statistics information. */
        void Print();

        /**
         *  Returns the isolation communication time of a particular operation for one iteration.
         *  @param opIdx operation's index
         *  @returns The time in CPU clock cyles
         */
        unsigned long long GetIsolationCommCycles(size_t opIdx);

        /**
         *  Returns the communication size of a particular operation.
         *  @param opIdx the operation index
         *  @returns The size value in bytes.
         */
        size_t GetCommSize(size_t opIdx);

        /**
         *  Returns the communication time of a particular operation.
         *  @param opIdx operation's index
         *  @returns The time in CPU clock cyles
         */
        unsigned long long GetCommCycles(size_t opIdx);

        /**
         *  Returns the compute time of a particular operation.
         *  @param opIdx operation's index
         *  @returns The time in CPU clock cyles
         */
        unsigned long long GetComputeCycles(size_t opIdx);

        /**
         *  Returns the total isolation communication time for all operations for one iteration.
         *  @returns The time in CPU clock cyles
         */
        unsigned long long GetTotalIsolationCommCycles();

        /**
         *  Returns the total communication size.
         *  @returns The size value in bytes.
         */
        size_t GetTotalCommSize();

        /**
         *  Returns the total communication time for all operations.
         *  @returns The time in CPU clock cyles.
         */
        unsigned long long GetTotalCommCycles();

        /**
         *  Returns the total compute time for all operations.
         *  @returns The time in CPU clock cyles.
         */
        unsigned long long GetTotalComputeCycles();
    };

    /**
     *  @brief A class to represent a collection of Operation objects with the same global mini-batch size
     */
    class Session
    {
        NO_EXPLICIT_CREATION(Session)

    public:
        /**
         * Sets the global mini-batch size. Must be called before any compute operation creation.
         * @param globalMinibatchSize the global mini-batch size
         */
        void SetGlobalMinibatchSize(size_t globalMinibatchSize);

        /** @returns The global mini-batch size. */
        size_t GetGlobalMinibatchSize();

        /** @returns The phase type. */
        PhaseType GetPhaseType();

        /**
         *  Creates an object containing the operation's registration information.
         *  @param opType the compute operation type
         *  @returns The operation's registration object.
         */
        OperationRegInfo* CreateOperationRegInfo(OpType opType);

        /**
         *  Deletes the previously created OperationRegInfo object.
         *  @param info the operation's registration object to delete
         */
        void DeleteOperationRegInfo(OperationRegInfo* info);

        /**
         *  Creates and adds an Operation object to the current session.
         *  @param info the operation's registration object that holds information about activations and/or parameter sets
         *  @param dist the distribution that will be used by the new operation (optional),
         *              can be set later with the Operation::SetDistribution() method
         */
        size_t AddOperation(OperationRegInfo* info, Distribution* dist = NULL);

        /** Removes all operations from the current session. */
        void RemoveOperations();

        /** @returns The number of operations for the current session. */
        size_t GetOperationCount();

        /**
         *  Returns the operation by index.
         *  @param idx the operation's index
         *  @returns The operation.
         */
        Operation* GetOperation(size_t idx);

        /** 
         *  Finalizes creation of the collection of Operations.
         *  Must be called after all operations are added to the session.
         *  Session-wide optimizations take place there.
         */
        void Commit();

        /**
         *  Returns the statistics information for the session
         *  @returns A Statistics class instance.
         */
        Statistics* GetStats();
    };

    /**
     *  @brief A singleton object that holds global Intel %MLSL functions
     */
    class Environment
    {
        NO_EXPLICIT_CREATION(Environment)

    public:

        /** @returns An Environment class instance. */
        static Environment& GetEnv();

        /** @returns The full Intel %MLSL API version. */
        static int GetVersion();

        /**
         *  An optional method to allow passing configuration information e.g. configuration file name.<br>
         *  For the MPI backend this may be NULL, but may be handy for the gRPC or ZMQ implementation.
         *  @param config a configuration string or the path to a configuration file (optional)
         */
        void Configure(const char* config = NULL);

        /**
         *  Initializes the library. Must precede any other library calls (except static methods of the Environment class).
         *  @param argc a pointer to the number of arguments
         *  @param argv an argument vector
         */
        void Init(int* argc, char** argv[]);

        /** Finalizes the library, cleans up and frees all the internally allocated memory. */
        void Finalize();

        /** @returns True if Init() has been called, false otherwise. */
        bool IsInitialized();

        /** @returns The global process index. */
        size_t GetProcessIdx();

        /** @returns The global number of processes. */
        size_t GetProcessCount();

        /**
         *  Creates a new Session object.
         *  @param phaseType the phase type (optional), can be used for internal optimizations
         *  @returns A session.
         */
        Session* CreateSession(PhaseType phaseType = PT_TRAIN);

        /**
         *  Deletes the previously created Session object.
         *  @param session the session to delete
         */
        void DeleteSession(Session* session);

        /**
         *  Creates a new Distribution object.
         *  @param dataPartitions the number of partitions for data (partitions on global mini-batch)
         *  @param modelPartitions the number of partitions for model (partitions on input activation)
         *  @returns A distribution.
         */
        Distribution* CreateDistribution(size_t dataPartitions, size_t modelPartitions);

        /**
         *  Creates a new Distribution object based on process colors passed. Don't use this method unless you absolutely need it.
         *  @param dataColor defines partitions on global mini-batch. Processes with the same dataColor get into the same data group
         *  @param modelColor defines partitions on input activation. Processes with the same modelColor get into the same model group
         *  @returns A distribution.
         */
        Distribution* CreateDistributionWithColors(int dataColor, int modelColor);

        /**
         *  Deletes the previously created Distribution object.
         *  @param distribution the distribution to delete
         */
        void DeleteDistribution(Distribution* distribution);

        /**
         *  Waits for a communication request to complete.
         *  @param req the communication request
         */
        void Wait(CommReq* req);

        /**
         *  Tests for completion of a communication request.
         *  @param req the communication request
         *  @param isCompleted the completion status of the request, true if request is completed, false otherwise
         */
        void Test(CommReq* req, bool* isCompleted);

        /**
         *  Intel %MLSL specific allocation function. Should be used to allocate communication buffers.
         *  Allocates size bytes and returns a pointer to the allocated memory.
         *  The memory address will be a multiple of alignment, which must be a power of two.
         *  @param size the number of bytes to allocate
         *  @param alignment the return pointer alignment
         *  @returns A pointer to the allocated memory, or NULL if the allocation fails.
         */
        void* Alloc(size_t size, size_t alignment);

        /**
         *  Intel %MLSL specific deallocation function. Frees the memory that was previously allocated with Alloc().
         *  @param ptr the pointer to the memory to be deallocated
         */
        void Free(void* ptr);

        /**
         * Sets quantization parameters.
         * Quantization parameters must be set before calling OperationRegInfo::AddParameterSet() with quantization enabled.
         * @param params the data structure containing information about the quantization library
         */
        void SetQuantizationParams(QuantParams* params);

        /**
         * @returns A pointer to the structure that contains information about quantization library.
         */
        QuantParams* GetQuantizationParams();
    };
};

#endif /* MLSL_HPP */
