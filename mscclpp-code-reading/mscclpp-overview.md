# MSCCL++ Overview

MSCCL++ is a redesign of the NCCL code. It decouples basic components in NCCL and exposes them as host-side or device-side interfaces. Host-side interfaces are invoked from CPU code, while device-side interfaces are invoked from GPU code. We give four examples of these basic components:

1. The GPU primitives that perform data transfer between GPUs are provided as device-side interfaces. Based on the primitive type, each interface is invoked by either a single GPU thread or a group of GPU threads.
2. The proxy FIFO mechanism enables the GPU to request assistance from the CPU to initiate RDMA transactions. It provides device-side interfaces for the GPU to post requests. It also provides host-side interfaces for the CPU to poll requests.
3.  The synchronization mechanism between the sender GPU and receiver GPU is exposed as device- and host-side interfaces. It coordinates data transfer.
4. The bootstrap network that exchanges peer info is exposed as host-side interfaces.

Most of these basic components are implemented like NCCL. You can think of them as clean-slate building blocks, where you can composite or stack them to realize arbitrary collective communication algorithms and patterns, to design your kernels with communication and computation co-optimization, and to replace NCCL with more controllable and understandable performance.

Thanks to MSCCL++, we don't have to reinvent the wheel. In this overview, I will guide you through a CUDA program that implements the All-Pairs All-Reduce algorithm using MSCCL++, allowing you to gain experience with MSCCL++. I will defer the implementation details of each MSCCL++ interface and requirements to subsequent notes.

## All-Pairs All-Reduce in MSCCL++

All-Pairs All-Reduce is a simple All-Reduce algorithm suitable for small messages and a single machine. Nevertheless, the following code is runnable on a cluster of machines of any size, as long as they are homogeneous. The idea is that each peer breaks its buffer into $W$ chunks, where $W$ is the world size. Peer $i$ is responsible for reducing the $i$-th chunk, so there are three steps to go:

1. Every peer first sends its local $i$-th chunk to a scratch buffer on Peer $i$
2. Peer $i$ reduces all peers' chunks into its $i$-th chunk by summing them together
3. Peer $i$ broadcasts its $i$-th chunk to all other peers.

> **Note 1:** By homogeneous, we mean that each machine has the same architecture and physical layout. That is, the number of GPUs and their NUMA affinity are the same. The number of InfiniBand (IB) devices is the same, and a railed network fabric interconnects them (i.e., the $i$-th IB card on one machine can access the $i$-th IB card on other machines). 
>
> **Note 2:** We will use the terms "RDMA" and "IB" interchangeably. We will use the terms "peer" and "rank" interchangeably.
>
> **Assumption:** We assume a one-to-one mapping between processes and GPUs. We only consider in-place All-Reduce (i.e., the output buffer equals the input buffer).

The CUDA code is as follows. Although it may seem lengthy, each part is structured and easy to understand. I will explain it to you step by step. The overall procedure is as follows.

1. Allocate the resources.
2. Set up full-mesh connections between all pairs of GPUs using the communicator.
3. Register local memory buffers, and send their handles to other peers so they can also access them (from their perspective, these are remote memory)
4. Build proxy channels between GPUs on different machines. Communication between them requires the assistance of the sender's and receiver's proxy threads.
5. Start proxy threads.
6. Build SM channels between GPUs on the same machine. The proxy thread is not involved in intra-node communication.
7. We test the host-side version of All-Pairs All-Reduce. That is, we only use the host-side interfaces for data communication.
8. We test the device-side version of All-Pairs All-Reduce. That is, we only use the device-side interfaces for data communication.
9. Once the two tests succeed, we release all CUDA and host-side resources.

```C++
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <vector>

#include <mpi.h>
#include <iostream>
#include <sstream>

#define CUDA_CHECK(cmd)                                                                                           \
  do {                                                                                                          \
    cudaError_t err = cmd;                                                                                      \
    if (err != cudaSuccess) {                                                                                   \
      std::string msg = std::string("CUDA failure: ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        " '" + cudaGetErrorString(err) + "'";                                                   \
      throw std::runtime_error(msg);                                                                            \
    }                                                                                                           \
  } while (0)

struct DataClass {
public:
    DataClass(int myRank, int totalRanks, int localRank, int nRanksPerNode,
              int numBuffers, int bufferSize): myRank(myRank), totalRanks(totalRanks), localRank(localRank), nRanksPerNode(nRanksPerNode),
              numBuffers(numBuffers), bufferSize(bufferSize) {
                  ibDeviceCount = mscclpp::getIBDeviceCount();
              }
    ~DataClass() {}

    void Intialize(std::string bootstrapIpPort) {
        // Set CUDA device
        CUDA_CHECK(cudaSetDevice(localRank));
        // Initialize host containers
        devicePtr.resize(numBuffers);
        localMemory.resize(numBuffers);
        remoteMemory.resize(numBuffers);
        proxyChannels.resize(numBuffers);
        proxyChannelDevicePtr.resize(numBuffers);
        smChannels.resize(numBuffers);
        smChannelDevicePtr.resize(numBuffers);
        // Initialize device memory
        for (int i = 0; i < numBuffers; i++) {
            devicePtr[i] = mscclpp::allocSharedCuda<int>(2 * bufferSize / sizeof(int));
        }
        for (int i = 0; i < numBuffers; i++) {
            proxyChannelDevicePtr[i] = mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::ProxyChannel>>(totalRanks);
        }
        for (int i = 0; i < numBuffers; i++) {
            smChannelDevicePtr[i] = mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(totalRanks);
        }
        // Initialize bootstrap and communicator
        bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, totalRanks);
        bootstrap->initialize(bootstrapIpPort);
        communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
        proxyService = std::make_shared<mscclpp::ProxyService>();
    }
    void Finalize() {
        // Device memory is automatically released when the last std::shared_ptr is destroyed
        // Release host memory
        connections.clear();
        devicePtr.clear();
        localMemory.clear();
        remoteMemory.clear();
        proxyChannels.clear();
        proxyChannelDevicePtr.clear();
        smChannels.clear();
        smChannelDevicePtr.clear();
        bootstrap.reset();
    }
    /**
     * From job launch
     */
    int myRank, totalRanks, localRank, nRanksPerNode;
    /**
     * From cmdline args
     */
    int numBuffers;
    int bufferSize;
    /**
     * From local machine
     */
    int ibDeviceCount;
    /**
     * From initialization
     */
    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
    std::shared_ptr<mscclpp::Communicator> communicator;
    std::shared_ptr<mscclpp::ProxyService> proxyService;
    /**
     * From runtime
     */
    // Connections
    std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections; // [i]: Connection to rank i
    // Buffers
    std::vector<std::shared_ptr<int>> devicePtr; // [i]: Buffer i's pointer (on device); the first half is for data, and the second half is for scratch
    std::vector<mscclpp::RegisteredMemory> localMemory; // [i]: Buffer i's registered memory
    std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory; // [i][j]: Rank j's Buffer i
    // Proxy channels
    std::vector<std::unordered_map<int, mscclpp::ProxyChannel>> proxyChannels; // [i][j]: Rank j's ProxyChannel i (associated with Buffer i)
    std::vector<std::shared_ptr<mscclpp::DeviceHandle<mscclpp::ProxyChannel>>> proxyChannelDevicePtr; // [i][j]: Rank j's ProxyChannel i's device handle on device
    // SM channels
    std::vector<std::unordered_map<int, mscclpp::SmChannel>> smChannels; // [i][j]: Rank j's SmChannel i (associated with Buffer i)
    std::vector<std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>>> smChannelDevicePtr; // [i][j]: Rank j's SmChannel i's device handle on device
};

// Map a rank to its local rank (assuming each node has the same numfer of ranks)
inline int RankToLocalRank(const DataClass &data, int rank) {
    return rank % data.nRanksPerNode;
}

// Map a rank to its node (assuming each node has the same number of ranks)
inline int RankToNode(const DataClass &data, int rank) {
    return rank / data.nRanksPerNode;
}

// Determine which IB device to use when communicating with a remote peer
inline mscclpp::Transport RankToIbDev(const DataClass &data, int rank) {
    mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                mscclpp::Transport::IB6, mscclpp::Transport::IB7};
    return IBs[(data.localRank + RankToLocalRank(data, rank)) % data.ibDeviceCount];
}

// Set up a full-mesh of connections between ranks
void SetupFullMeshConnections(DataClass &data) {
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures(data.totalRanks);
    for (int j = 0; j < data.totalRanks; j++) {
      if (j != data.myRank) {
        if ((RankToNode(data, j) == RankToNode(data, data.myRank))) {
          connectionFutures[j] = data.communicator->connectOnSetup(j, 0, mscclpp::Transport::CudaIpc);
        } else {
          connectionFutures[j] = data.communicator->connectOnSetup(j, 0, RankToIbDev(data, j));
        }
      }
    }
    data.communicator->setup();
    for (int j = 0; j < data.totalRanks; j++) {
      if (j != data.myRank) {
        data.connections[j] = connectionFutures[j].get();
      }
    }
}

// Register each local memory to all other ranks
void RegisterMemoryPairs(DataClass &data) {
    mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;
    for (int j = 0; j < data.totalRanks; j++) {
        if (RankToNode(data, j) != RankToNode(data, data.myRank)) {
            transport |= RankToIbDev(data, j);
        }
    }
    for (int i = 0; i < data.numBuffers; i++) {
        // Register each buffer with the communicator
        data.localMemory[i] = data.communicator->registerMemory(data.devicePtr[i].get(), 2 * data.bufferSize, transport);
        // Send local memory to all remote ranks
        std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemories;
        for (int j = 0; j < data.totalRanks; j++) {
            if (j != data.myRank) {
                data.communicator->sendMemoryOnSetup(data.localMemory[i], j, /*tag*/ 0);
                futureRemoteMemories[j] = data.communicator->recvMemoryOnSetup(j, /*tag*/ 0);
            }
        }
        data.communicator->setup();
        // Receive remote memories
        for (int j = 0; j < data.totalRanks; j++) {
            if (j != data.myRank) {
                data.remoteMemory[i][j] = futureRemoteMemories[j].get();
            }
        }
    }
}

// Build proxy channels to each remote peer for each buffer
void BuildProxyChannels(DataClass &data) {
    for (int i = 0; i < data.numBuffers; i++) {
        for (int j = 0; j < data.totalRanks; j++) {
            if (RankToNode(data, j) != RankToNode(data, data.myRank)) {
                mscclpp::SemaphoreId sid = data.proxyService->buildAndAddSemaphore(*data.communicator, data.connections[j]);
                mscclpp::MemoryId dstid = data.proxyService->addMemory(data.remoteMemory[i][j]);
                mscclpp::MemoryId srcid = data.proxyService->addMemory(data.localMemory[i]);
                data.proxyChannels[i].emplace(j, data.proxyService->proxyChannel(sid, dstid, srcid));
            }
        }
        data.communicator->setup();
        // Create device handles on device
        std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>> deviceHandles(data.totalRanks);
        for (int j = 0; j < data.totalRanks; j++) {
            if (RankToNode(data, j) != RankToNode(data, data.myRank)) {
                deviceHandles[j] = data.proxyChannels[i].at(j).deviceHandle();
            }
        }
        mscclpp::memcpyCuda<mscclpp::DeviceHandle<mscclpp::ProxyChannel>>(
                data.proxyChannelDevicePtr[i].get(), deviceHandles.data(), data.totalRanks, cudaMemcpyHostToDevice);
    }
}

// Build shared memory channels to each local peer for each buffer
void BuildSmChannels(DataClass &data) {
    for (int i = 0; i < data.numBuffers; i++) {
        for (int j = 0; j < data.totalRanks; j++) {
            if (j != data.myRank && RankToNode(data, j) == RankToNode(data, data.myRank)) {
                std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore> semaphore =
                        std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*data.communicator, data.connections[j]);
                data.smChannels[i].emplace(j, mscclpp::SmChannel(semaphore, data.remoteMemory[i][j], data.localMemory[i].data(), nullptr));
            }
        }
        data.communicator->setup();
        // Create device handles on device
        std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> deviceHandles(data.totalRanks);
        for (int j = 0; j < data.totalRanks; j++) {
            if (j != data.myRank && RankToNode(data, j) == RankToNode(data, data.myRank)) {
                deviceHandles[j] = data.smChannels[i].at(j).deviceHandle();
            }
        }
        mscclpp::memcpyCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(
                data.smChannelDevicePtr[i].get(), deviceHandles.data(), data.totalRanks, cudaMemcpyHostToDevice);
    }
}

inline int InitVal(int buffer, int rank, int index) {
    return rank + buffer * 121 + index * 11;
}

inline int AllReduceVal(int buffer, int totalRanks, int index) {
    int result = 0;
    for (int j = 0; j < totalRanks; j++) {
        result += InitVal(buffer, j, index);
    }
    return result;
}

inline void DeviceBufferInit(DataClass &data) {
  size_t dataCount = data.bufferSize / sizeof(int);
  for (int i = 0; i < data.numBuffers; i++) {
    std::vector<int> hostBuffer(dataCount, 0);
    for (size_t j = 0; j < dataCount; j++) {
      hostBuffer[j] = InitVal(i, data.myRank, j);
    }
    mscclpp::memcpyCuda<int>(data.devicePtr[i].get(), hostBuffer.data(), dataCount, cudaMemcpyHostToDevice);
  }
}

static std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    auto printUsage = [](const char* prog) {
        std::stringstream ss;
        ss << "Usage: " << prog << " [-n numBuffers] [-b bufferSize] -a bootstrapIpPort [-h]\n"
           << "Options:\n"
           << "  -n numBuffers       Number of buffers to use (default: 5)\n"
           << "  -b bufferSize       Size of each buffer in bytes (default: 1048576)\n"
           << "  -a bootstrapIpPort  IP and port for bootstrap communication\n"
           << "  -h                  Show this help message\n";
        std::cout << ss.str();
    };
    // Default values
    std::unordered_map<std::string, std::string> args;
    args["n"] = "5"; // numBuffers
    args["b"] = "1048576"; // bufferSize
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            args["n"] = argv[++i];
        } else if (arg == "-b" && i + 1 < argc) {
            args["b"] = argv[++i];
        } else if (arg == "-a" && i + 1 < argc) {
            args["a"] = argv[++i];
        } else if (arg == "-h") {
            printUsage(argv[0]);
            args["h"] = "";
        }
    }
    return args;
}

// This kernel should be launched with 1 block
__global__ void HostToDeviceAllPairsAllReduceKernel(
        int *ptr,
        int scratchOffsetInInt,
        int chunkSizeInInt,
        int myRank, int totalRanks) {
    int *output = ptr + myRank * chunkSizeInInt;
    int *scratchPtr = ptr + scratchOffsetInInt;
    for (int j = 0; j < totalRanks; j++) {
        if (j != myRank) {
            int *input = scratchPtr + j * chunkSizeInInt;
            // Perform all-reduce on the chunk
            for (int i = threadIdx.x; i < chunkSizeInInt; i += blockDim.x) {
                output[i] += input[i];
            }
        }
    }
}

__global__ void DeviceToDeviceAllPairsAllReduceKernel(
        mscclpp::DeviceHandle<mscclpp::ProxyChannel> *proxyChannels,
        mscclpp::DeviceHandle<mscclpp::SmChannel> *smChannels,
        int *ptr,
        int scratchOffsetInInt,
        int chunkSizeInInt,
        int myRank,
        int totalRanks,
        int nRanksPerNode) {
    int scratchOffsetInBytes = scratchOffsetInInt * sizeof(int);
    int chunkSizeInBytes = chunkSizeInInt * sizeof(int);
    for (int j = 0; j < totalRanks; j++) {
        // Sends out chunks via channels
        if((j / nRanksPerNode) != (myRank / nRanksPerNode)) {
            mscclpp::DeviceHandle<mscclpp::ProxyChannel> &channel = proxyChannels[j];
            if (threadIdx.x == 0) {
                channel.putWithSignal(scratchOffsetInBytes + myRank * chunkSizeInBytes, j * chunkSizeInBytes, chunkSizeInBytes);
            }
        } else if (j != myRank) {
            mscclpp::DeviceHandle<mscclpp::SmChannel> &channel = smChannels[j];
            channel.put<16, true>(scratchOffsetInBytes + myRank * chunkSizeInBytes, j * chunkSizeInBytes, chunkSizeInBytes, threadIdx.x, blockDim.x);
            if (threadIdx.x == 0) {
                channel.signal();
            }
        }
    }

    // Thread 0 wait for all chunks to be received
    for (int j = 0; j < totalRanks; j++) {
        if ((j / nRanksPerNode) != (myRank / nRanksPerNode)) {
            mscclpp::DeviceHandle<mscclpp::ProxyChannel> &channel = proxyChannels[j];
            if (threadIdx.x == 0) {
                channel.wait();
            }
        } else if (j != myRank) {
            mscclpp::DeviceHandle<mscclpp::SmChannel> &channel = smChannels[j];
            if (threadIdx.x == 0) {
                channel.wait();
            }
        }
    }

    // Let other threads wait for thread 0
    __syncthreads();

    // All-reduce chunks
    for (int j = 0; j < totalRanks; j++) {
        if (j != myRank) {
            for (int i = threadIdx.x; i < chunkSizeInInt; i += blockDim.x) {
                ptr[myRank * chunkSizeInInt + i] += ptr[scratchOffsetInInt + j * chunkSizeInInt + i];
            }
        }
    }

    // Sends back chunks via channels
    for (int j = 0; j < totalRanks; j++) {
        if ((j / nRanksPerNode) != (myRank / nRanksPerNode)) {
            mscclpp::DeviceHandle<mscclpp::ProxyChannel> &channel = proxyChannels[j];
            if (threadIdx.x == 0) {
                channel.putWithSignal(myRank * chunkSizeInBytes, myRank * chunkSizeInBytes, chunkSizeInBytes);
            }
        } else if (j != myRank) {
            mscclpp::DeviceHandle<mscclpp::SmChannel> &channel = smChannels[j];
            channel.put<16, true>(myRank * chunkSizeInBytes, myRank * chunkSizeInBytes, chunkSizeInBytes, threadIdx.x, blockDim.x);
            if (threadIdx.x == 0) {
                channel.signal();
            }
        }
    }

    // Wait for all chunks to be received
    for (int j = 0; j < totalRanks; j++) {
        if ((j / nRanksPerNode) != (myRank / nRanksPerNode)) {
            mscclpp::DeviceHandle<mscclpp::ProxyChannel> &channel = proxyChannels[j];
            if (threadIdx.x == 0) {
                channel.wait();
            }
        } else if (j != myRank) {
            mscclpp::DeviceHandle<mscclpp::SmChannel> &channel = smChannels[j];
            if (threadIdx.x == 0) {
                channel.wait();
            }
        }
    }
}

// All-pairs all-reduce via connection's write() and flush() methods
// Algorithm logic:
//   Rank i divides its 1st buffer into totalRanks chunks, and is responsible for allreducing Chunk i.
//   It sends Chunk j to Rank j, and waits for the all-reduced Chunk j from it.
//   It collects Chunk i from all other ranks, and sends them the all-reduced chunk.
//   We use the 2nd buffer as the scratchpad
void TestHostToDeviceAllPairsAllReduce(DataClass &data) {
    DeviceBufferInit(data);
    // Prevent data hazards (write after read)
    data.communicator->bootstrap()->barrier();
    // Sends out chunks
    int chunkSizeInBytes = data.bufferSize / data.totalRanks;
    int scratchOffsetInBytes = data.bufferSize;
    int buffer = 0;
    for (int j = 0; j < data.totalRanks; j++) {
        if (j != data.myRank) {
            data.connections[j]->write(data.remoteMemory[buffer][j], scratchOffsetInBytes + data.myRank * chunkSizeInBytes, data.localMemory[buffer],
                        j * chunkSizeInBytes, chunkSizeInBytes);
        }
    }
    for (int j = 0; j < data.totalRanks; j++) {
        if (j != data.myRank) {
            data.connections[j]->flush();
        }
    }
    // Wait for all chunks to be received
    data.communicator->bootstrap()->barrier();
    // All-reduce chunks
    HostToDeviceAllPairsAllReduceKernel<<<1, 1024>>>(data.devicePtr[buffer].get(), scratchOffsetInBytes / sizeof(int),
            chunkSizeInBytes / sizeof(int), data.myRank, data.totalRanks);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Sends back chunks
    for (int j = 0; j < data.totalRanks; j++) {
        if (j != data.myRank) {
            data.connections[j]->write(data.remoteMemory[buffer][j], data.myRank * chunkSizeInBytes, data.localMemory[buffer],
                        data.myRank * chunkSizeInBytes, chunkSizeInBytes);
        }
    }
    for (int j = 0; j < data.totalRanks; j++) {
        if (j != data.myRank) {
            data.connections[j]->flush();
        }
    }
    // Wait for all chunks to be sent
    data.communicator->bootstrap()->barrier();
    // Verify results
    std::vector<int> hostBuffer(data.bufferSize / sizeof(int), 0);
    mscclpp::memcpyCuda<int>(hostBuffer.data(), data.devicePtr[buffer].get(), data.bufferSize / sizeof(int), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < hostBuffer.size(); i++) {
        if (hostBuffer[i] != AllReduceVal(buffer, data.totalRanks, i)) {
            throw std::runtime_error("Data mismatch at rank " + std::to_string(data.myRank) +
                                 " buffer " + std::to_string(buffer) + " index " + std::to_string(i) +
                                 ": expected " + std::to_string(AllReduceVal(buffer, data.totalRanks, i)) +
                                 ", got " + std::to_string(hostBuffer[i]));
        }
    }
}

void TestDeviceToDeviceAllPairsAllReduce(DataClass &data) {
    DeviceBufferInit(data);
    // // Prevent data hazards (write after read)
    // data.communicator->bootstrap()->barrier();
    // Sends out chunks
    int chunkSizeInBytes = data.bufferSize / data.totalRanks;
    int chunkSizeInInt = chunkSizeInBytes / sizeof(int);
    int buffer = 0;
    DeviceToDeviceAllPairsAllReduceKernel<<<1, 1024>>>(
            data.proxyChannelDevicePtr[buffer].get(),
            data.smChannelDevicePtr[buffer].get(),
            data.devicePtr[buffer].get(), data.bufferSize / sizeof(int),
            chunkSizeInInt, data.myRank, data.totalRanks, data.nRanksPerNode);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Verify results
    std::vector<int> hostBuffer(data.bufferSize / sizeof(int), 0);
    mscclpp::memcpyCuda<int>(hostBuffer.data(), data.devicePtr[buffer].get(), data.bufferSize / sizeof(int), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < hostBuffer.size(); i++) {
        if (hostBuffer[i] != AllReduceVal(buffer, data.totalRanks, i)) {
            throw std::runtime_error("Data mismatch at rank " + std::to_string(data.myRank) +
                                 " buffer " + std::to_string(buffer) + " index " + std::to_string(i) +
                                 ": expected " + std::to_string(AllReduceVal(buffer, data.totalRanks, i)) +
                                 ", got " + std::to_string(hostBuffer[i]));
        }
    }
}

int main(int argc, char* argv[]) {
    int myRank, totalRanks, localRank, nRanksPerNode;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_size(shmcomm, &nRanksPerNode);
    MPI_Comm_rank(shmcomm, &localRank);
    MPI_Comm_free(&shmcomm);

    // Parse argument
    int numBuffers, deviceBufferSize;
    std::string bootstrapIpPort;
    auto args = parseArgs(argc, argv);
    if (args.find("h") != args.end()) { // help message
        MPI_Finalize();
        return 0;
    }
    try {
        numBuffers = std::stoi(args["n"]);
        deviceBufferSize = std::stoi(args["b"]);
        if (numBuffers <= 0 || deviceBufferSize <= 0) {
            throw std::invalid_argument("numBuffers and bufferSize must be positive integers.");
        }
        // if (numBuffers < 2) {
        //     throw std::invalid_argument("numBuffers must be at least 2.");
        // }
        deviceBufferSize = (deviceBufferSize + totalRanks * 16 - 1) / (totalRanks * 16) * (totalRanks * 16); // Round up to nearest multiple of totalRanks * 16
        if (args.find("a") != args.end()) {
            bootstrapIpPort = args["a"];
        } else {
            throw std::invalid_argument("-a is required.");
        }
        if (bootstrapIpPort.find(':') == std::string::npos) {
            throw std::invalid_argument("-a must be in the format 'ip:port'.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialize data class
    DataClass data(myRank, totalRanks, localRank, nRanksPerNode, numBuffers, deviceBufferSize);
    try {
        data.Intialize(bootstrapIpPort);
    } catch (const std::exception& e) {
        std::cerr << "Error during initialization: " << e.what() << std::endl;
        data.Finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Prepare for the collective operation
    SetupFullMeshConnections(data);
    RegisterMemoryPairs(data);
    BuildProxyChannels(data);
    BuildSmChannels(data);

    // Start proxy service
    data.proxyService->startProxy();

    try {
        TestHostToDeviceAllPairsAllReduce(data);
        TestDeviceToDeviceAllPairsAllReduce(data);
    } catch (const std::exception& e) {
        std::cerr << "Error during all-pairs all-reduce: " << e.what() << std::endl;
        data.proxyService->stopProxy();
        data.Finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    std::cout << "Rank " << myRank << " finished successfully." << std::endl;
    data.proxyService->stopProxy();
    data.Finalize();
    MPI_Finalize();
}
```

## Before Starting: DataClass (a Helper class)

To facilitate the mangement of multiple resources and peer info, we collect all the resource handles and peer info in a `DataClass`. This class has nothing to do with MSCCL++. It just helps a more organized code design. Any design presented in this section is application-specific, i.e., unrelated to MSCCL++. You can customize your own version.

We maintain three kinds of info, which are constant once initialized:

1. (Line 72) Rank-related info: Since we launch the job using MPI, such info are obtained from MPI directly.
2. (Lines 76-77) Buffer size and number of buffers per rank: These two are command line arguments. Although we only use Buffer 0 in both tests, we still allow multiple buffers to showcase a general usage. 
3. (Line 81) Number of IB devices on a machine.

We maintain five types of resources:

1. (Lines 85-87) Shared pointers to MSCCL++ bootstrap network, communicator, and proxy service. We always use C++ smart pointers when possible to lift the burden of host and device memory management. These objects are all host-resident.

2. (Line 92) My connections to other peers. `connections[i]` means the connection to Peer `i`. We store them in a  `std::unordered_map`  because we want to skip an index of my rank. These objects are all host-resident.

3. (Lines 94-96) Pointers to buffers. All buffers are allocated on GPU. There are three kinds of buffers:

   - `devicePtr[i]` points to the start of Buffer `i` on my GPU. Its type is `std::shared_ptr<int>`. Its value (`int *`) is a virtual address registered in the host process's space that will be translated to a GPU address when dereferencing.
   - `localMemory[i]` has type `mscclpp::RegisteredMemory`. It points to the same buffer as `devicePtr[i]`. We get it by registering the buffer to the communicator. It contains more information than `devicePtr[i]`, like the transport that can be used on this memory (e.g., CUDA IPC and the IB devices).
   - `remoteMemory[i][j]` is the registered memory for Buffer `i` of Rank `j`. When we access a address in `remoteMemory[i][j]`, we effectively access a remote memory. It also has type `mscclpp::RegisteredMemory`, and is the result of memory registration process.

   Each buffer has twice the size that the user specified (Line 77). We use the first half as data buffer, and the second half as the scratch buffer.

4. (Lines 98-99) Pointers to proxy channels. Each proxy channel is associated with a connection, a local buffer, and a remote buffer. Therefore, `proxyChannels[i][j]` is the channel of  between me and Rank `j` on Buffer `i`. Data transfer between my Buffer `i` and Rank `j`'s Buffer `i` should use this channel. On the other hand, `proxyChannelDevicePtr[i][j]` points to a device-resident handle of the proxy channel that provides the device-side primtives for GPU to invoke (e.g., `put()`, `signal()`, `flush()`, and `wait()`).

5. (Lines 101-102) Pointers to SM channels. Each SM channel is also associated with a connection, a local buffer, and a remote buffer. The only difference is that SM channel is for intra-node GPU communication, while proxy channel is for inter-node GPU communication. The device handle for SM channel between me and Rank `j` on Buffer `i` is `SmChannels[i][j]`.

> **Note:** Channels in MSCCL++ are different from NCCL. MSCCL++ channel is a connection between two buffers on two GPUs. NCCL channel is a ordering of intra-node GPUs that dictates the flow of data inside a machine.

We also provide three helpful functions.

1. (Lines 106-108) `RankToLocalRank()` converts a rank to its local rank. We assume each machine host the same number of ranks.
2. (Lines 111-113) `RankToNode()` converts a rank to its resident machine.
3. (Lines 116-121) `RankToIbDev()` assigns an IB device between me and another rank. Its logic ensures that a connection always goes through IB devices at the same index on both ends.

## 1. Allocate the Resources

We first fetch rank info from MPI (Lines 465-473). We get IB device number from `mscclpp::getIBDeviceCount()` (which is just a wrapper of `ibv_get_device_list()`) (Line 25). Now we allocate resources by `DataClass::Initialize()`.

First, we set CUDA device based on the local rank (Line 31). Then, we resize the host-side containers to their proper sizes (Lines 33-39). Then, we allocate GPU buffers using `mscclpp::allocSharedCuda<T>()` (Lines 42, 45, 48), which returns a `std::shared_ptr<T>`. Compared to directly calling `cudaMalloc()` and storing `T *`, the benefit of this method is that this smart pointer uses a MSCCL++-defined deleter. **This deleter automically calls `cudaFree()` when no more smart pointers point to the GPU buffer.** Therefore, we don't have to call `cudaFree()` explicitly. The C++ smart pointer system will handle the hassle for us.

Next, we initialize the bootstrap network, communicator, and proxy service (Lines 51-54). Their constructors do nothing more than store the arguments. The only nontrivial call `boostrap->initialize(bootstrapIpPort)` creates a fully functional bootstrap network among involved processes using the same procedure as in NCCL. We defer the details to [<font color="red">A later note</font>]().

## 2. Set up Full-mesh Connections

Function `SetupFullMeshConnections()` contains the corresponding logic (Lines 124-141). **Connection setup is an async process.** Specifically, `mscclpp::NonblockingFuture` (Line 125) wraps arround `std::future`, which represents an async operation (might be running in a separate thread). To obtain the final result, you use its `get()` method. If the result is not yet available, `get()` blocks the calling thread until it is.

**Connection setup is also a two-sided process.** That is, to build a connection between GPU `i` and `j`, Rank `i` should call `Communicator::connOnSetup()` to `j`, and Rank `j` should also call `Communicator::connOnSetup()` to `i`. They return aysnc objects, whose execution is delayed until `Communicator::setup()` (Line 135). During execution, the two ends of a connection exchange their endpoint information like the host hash. These messages go over the boostrap network.

The last argument in `Communicator::connOnSetup()` is the transport method. This should be `mscclpp::Transport::CudaIpc` for intra-node connection, and a specific IB device for inter-node connection. **Intra-node connection relies on GPU peer access for data transfer, and inter-node connection relies on GDR for data transfer.** The latter will create an RDMA CQ and QP on the associated IB device during `Communicator::connOnSetup()`. This QP info is also exchanged between the two ends during the async execution. At runtime, data transfer over an inter-node connection goes through its associated IB card.

> **Note:** GPU Direct RDMA (GDR) means that RDMA NIC (or IB device) can directly access the GPU's memory by targeting its PCIe address, bypassing the CPU and host memory for faster data transfers.

## 3. Buffer Registration

Function `RegisterMemoryPairs()` contains the corresponding logic (Lines 144-170). Registering the local Buffer $i$ will set up correct context related to that buffer in order to be used by the communicator. Registering the remote Buffer $i$ of Rank $j$ enforces Rank $j$ to generate and share the right handles of its Buffer $i$ so the current process can access it.

We register a local buffer via `Communicator::registerMemory()` (Line 153). The last argument contains all the possible transports that can be used with this buffer.

1. If CUDA IPC is included, the buffer is ready to be accessed by `cudaIpcGetMemHandle()`. This local handle can then be shared to other processes on the same machine to let them access the buffer.
2. For each IB device contained, the buffer will be registered as an MR in that device's PD. This registered MR contains a rkey (remote key). A remote peer can use this rkey to gain permissions to the buffer. It is fine that one buffer is registered to multiple IB devices, as long as the programmer takes care of avoiding any data hazard.

The returned value is a `RegisteredMemory`, which **does not** own the memory it points to. It just provides a way to transfer metadata about the memory to other processes.

Next, we register the remote buffer. **Registering remote buffer is also a two-sided and async operation.** That is, to register Rank $j$'s Buffer $i$ to Rank $k$, Rank $j$ should call `Communicator::sendMemoryOnSetup()`, and Rank $k$ should call `Communicator::recvMemoryOnSetup()`. Both functions generate async objects, so we need a `Communicator::setup()` to trigger async execution. We only need the result of async execution of `recvMemoryOnSetup()`, which is the remote buffer's `RegisteredMemory`. This `RegisteredMemory` contains the right handles to access the remote memory.

1. If the transport contains CUDA IPC and the buffer is on the same machine (judged by host hash in the `RegisteredMemory`), it opens the imported IPC handle to get a usable one. This handle registers a virtual address in the local GPU, which is translated to accessing the peer GPU when dereferencing.
2. If the transport contains IB device, it stores the MR info, including the rkey. Different from CUDA IPC, it does not correspond to a vritual address in either GPU's or CPU's address space.

Handles of a registered memory is also exchanged over the bootstrap network.

## 7. Host-side All-Reduce

Now we have everything needed by host-side All-Reduce at hand (Lines 385-435). Although for the second test (device-side All-Reduce), more preparation is needed. Function `TestHostToDeviceAllPairsAllReduce()` contains the corresponding logic of the first test (Lines 385-435). 

First, we initialize the content of all the local data buffers (Line 386), although we will only use Buffer 0. `Connection`s that we set up in Step 2 provide host-side interfaces for data communication, namely `write()` and `flush()`.

- **Connection's `write()` is a one-sided operation**. That is, only the sender should call `write()` to accomplish a data transfer; the receiver does not issue any call. It will use the associated transport of that connection to transfer data.
- **Connection's `flush()` waits until all preceding `write()` are done.** Note that `flush()` only works for the sender. Readers may wonder how to synchronize between the sender and receiver. The answer is semaphore. For now, this test simply uses a global barrier `bootstrap->barrier()` to synchronize all processes.

After buffer initialization, Rank $k$ sends out Chunk $j$ (in the data buffer) to Peer $j$'s Chunk $k$ (in the scratch buffer) using `Connection::write()` (Lines 395-396). It then waits for all the `write()` to finish by calling a corresponding `flush()` for each `write()` (Line 401).

The next thing to do in Rank $k$ is reduce all chunks except for $k$ in the scratch buffer into Chunk $k$ in the data buffer. We launch a hand-written kernel to do this (Lines 407-408). **But before launch kernel, a global barrier is needed (Line 405) because we must ensure the process has received all the chunks from other processes.**

> **Caution:** Data hazards would occur without this barrier since `flush()` only works for senders. Suppose we have two processes. Rank 0 sends to Rank 1 its Chunk 1, and Rank 1 sends its Chunk 0 to Rank 0. Chances are that Rank 0 has done sending Chunk 1 (i.e., returning from `flush()`), but Rank 1 is halfway sending Chunk 0 (i.e., blocking on `flush()`). Now, if Rank 0 launches the kernel, the Chunk 0 in its scratch buffer is not ready! Rank 0 must wait until Rank 1 returns from `flush()`. This explains why we use a global barrier.

The launched kernel is simple (Lines 275-291). It only contains 1 thread block, where a thread iteratively accesses $W-1$ chunks in the scratch buffer and add it to the chunk in the data buffer ($W$ is the world size). Since kernel launch is async w.r.t. CPU, we must call `cudaDeviceSynchrnoize()` (Line 409) to ensure the reduction is done before we can send back the reduced chunk to other processes.

Up to this point, Rank $k$ has the reduced Chunk $k$ in its data buffer. The last work is to write it to all other processes' Chunk $k$. Similarly, we use a `write()` (Line 413) and a corresponding `flush()` (Line 419). **Again, we need a global barrier to ensure the current processes have received all the chunks from other processes (Line 423).** Now, we are ready to verify the results. We omit the verification process for brevity.

> **Note:** We need at least 2 global barriers in the host-side All-Reduce.

## 4. Build Proxy Channels

Function `BuildProxyChannels()` contains the corresponding logic (Lines 173-194). Each proxy channel is associated with a semaphore, a (registered) local buffer, and a (registered) remote buffer. A semaphore is in turn associated with a connection we build in Step 2. **Semaphore provides the synchronization mechanism between a sender and a receiver.**

**Proxy channels are important because they provide device-side interfaces.** In MSCCL++, all device-side objects can be obtained from their host-side counterparts via the `deviceHandle()` method. A device handle only contains CUDA pointers, GPU data, and methods that are only invocable from GPU code. For example, the host-side proxy channel is the class `ProxyChannel`, and we can obtain its device handle `DeviceHandle<ProxyChannel>` via `ProxyChannel::deviceHandle()`.

Go back to the function. Rank $k$ iterates over all peers that are not on the same machine (Line 176). **We first build a semaphore via `ProxyService::buildAndAddSemaphore()` (Line 177).** Its arguments specify the communicator and connection associated with the semaphore. **Building semaphore is a two-sided and async process. The reason is that we need to set up a small counter and maintain its local and remote values, which means we have to expose the local counter's address to remote ranks.** Following the two-sided async buffer registration process, building semaphore is also two-sided and async. We call the corresponding `Communicator::setup()` in Line 183.

**Next, we create a proxy channel via `ProxyService::proxyChannel`, whose arguments specify the semaphore and the two associated buffers (Line 180).** Therefore, each buffer is contained in $W$ proxy channels in our full-mesh setup. After setting up semaphores, we can obtain device handles for the created proxy channels (Line 188). We organize all the Buffer $i$'s proxy channels as a $W$-sized array (indices corresponding to local ranks are left empty). We then copy this array to GPU via `mscclpp::memcpyCuda()` (Lines 191-192), which just wraps around `cudaMemcpy()`.

**Whenever GPU issues a `put()`, `signal()`, `flush()`, or `wait()` request, proxy channels' device handle posts a corresponding request (along with the request arguments) into a FIFO queue.** A host-side per-rank proxy thread continually polls this FIFO queue, fetches the tail request, executes the requests, and dequeues the tail. For example, for a `put()` request, the proxy thread will post a RDMA WR into QP. For a `flush()` request, the proxy thread will polls the CQ for each preceding WR's CQE (Completion Queue Element). Any non-successful status will be reported.

## 5. Build SM Channels

Function `BuildSmChannels()` contains the corresponding logic (Lines 197-217). Similar to building proxy channels, we first construct a semaphore (Lines 201-202) and then construct a SmChannel from this semaphore, a local buffer, and a remote buffer (Line 203). Since we do not need the help of `ProxyService`, the invoked functions can be different, but the overall idea remains the same. For example, we still have to call `Communicator::setup()` (Line 206) to execute async operations required by semaphore construction.

Likewise, `SmChannel` provides device-side interfaces via its device handle (Line 211). We organize all the Buffer $i$'s SM channels as a $W$-sized array (indices corresponding to remote ranks are left empty). At runtime, if we need cross-node communication, we should use the proxy channel; otherwise, we should use the SM channel.

## 6. Start Proxy Threads

We start a per-rank host-side proxy thread using `ProxyService::startProxy()` (Line 525). This starts the default MSCCL++ proxy thread, which keeps polling the FIFO queue and processing the tail request. Users can define customized request handling logic and the proxy thread, and we defer details to [<font color="red">A later note</font>](). The proxy thread stops when the user calls `ProxyService::stopProxy()` (Line 538). This FIFO queue was created when we initialize `ProxyService`. Therefore, all requests from a rank (may be to different destinations) will be enqueued to the same FIFO. Proxy thread will execute them in their queueing order.

## 8. Device-side All-Reduce

Now we are to ready to proceed to the second test, whose All-Pairs All-Reduce logic are embedded in the kernel function `DeviceToDeviceAllPairsAllReduceKernel()` (Lines 293-377). Before launching the kernel (Line 445) We initialize the buffer in the same manner as in the first test (Line 438).

The first job of Rank $k$ is to sends out its local Chunk $j(\neq k)$ in the data buffer to Rank $j$'s Chunk $k$ in the scratch buffer. We use a channel's `put()` method (Lines 309, 313). **Note that just like `Connection::write()`, `put()` is a one-sided function, so there is no receiver-side call.** A major difference between the proxy channel's and the SM channel's `put()` is that **the former is called by only one GPU thread, while the latter is called by a collection of GPU threads.** This is because proxy channel only pushes a request into the FIFO, while SM channel uses thread-copy. After each `put()` we usually call a `signal()`, which increments an internal counter in the semaphore, and pushes its new value to the other side of the channel using the same mechanism as data transfer (i.e., FIFO for proxy channel, and thread-copy for SM channel). For a proxy channel, this can be done in one function `putWithSignal()` (Line 309). For a SM channel, there is no such thing since `signal()` is called by one GPU thread.

The receiver side calls `wait()` to wait for preceding transactions in a channel to finish (Lines 325, 330). It increments its expected value and waits until the value sent from the other side reaches this expected value. **Therefore, the receiver's `wait()` has a one-to-one correspondence to the sender's `signal()`.** Like `signal()`, `wait()` is called by one GPU thread for both types of channels.

Now we are about to reduce chunks. **But since only one thread waits for the transaction to finish, we use a block-level synchrnoization primitive `__syncthreads()` (Line 336) to let other threads wait for Thread 0.** Once all threads reach this barrier, all chunks in the scratch buffers are ready. The reduce logic (Lines 339-345) is the same as before, so we will omit it.

It's time for Rank $k$ to send back its reduced Chunk $k$ in the data buffer to all other ranks' Chunk $k$. Again we use the `putWithSignal()` for proxy channels, and `put()` for SM channels. Thread 0 waits for all reduced chunks to be reduced.

We can verify the result now. **But since kernel launch is async w.r.t. CPU, we need a `cudaDeviceSynchronize()` to wait for the kernel to finish (Line 450).** This host-device synchronization also has the effect of a block-level barrier, because all other threads must wait for Thread 0 to finish before the kernel ends.

> **Note:** We need no global barrier and only 1 block-level barrier in the device-side All-Reduce

## 9. Resource Dealloction

We first stops proxy thread (Line 538), and uses `DataClass::Finalize()` method to release host- and device-side resources. Recall that the smart pointer takes care of `cudaFree()`. Finally, we finalize MPI with `MPI_Finalize()` (Line 540).

## Wrap up: MSCCL++ Abstraction

In this overview, we have used the following abstractions provided by MSCCL++:

- **Connection:** Memory-independent; Relying on a specific tranport;
- **Registered memory:** Does not own the memory; Contains metadata for the remote side to access that memory or for me to access a remote memory; One memory can be accessed via multiple transports
- **Semaphore:** Synchronization mechanism between a sender and a receiver; Relying on a connection
- **Proxy channels:** Relying on a semaphore, a local buffer, and a remote buffer (on a different machine);
- **SM channels:** Relying on a semaphore, a local buffer, and a remote buffer (on the same machine);
- **FIFO:** A queue where GPU pushes requests, and the proxy thread executes the requests
- **Proxy thread:** Handling requests from a GPU; Per-rank;

In subsequent notes, we will delve into their implementation to deepen our understanding. Note that there are some procedures in NCCL missing in MSCCL++, like the channel search (i.e., organization of local GPUs), and the tuning model (i.e., selecting a suitable algorithm and protocol based on a collective operation and message size). Users have to customize their own versions if needed.
