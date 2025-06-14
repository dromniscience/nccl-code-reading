# MSCCL++ Deep Dive 02: Communicator

## Building a Functional Communicator

After initializing the bootstrap network, we build the communicator. This requires multiple steps.

1. Setting up connections between pairs of processes
2. Registering buffers pertinent to each connection
3. Creating semaphores for sender-receiver synchronization
4. Establishing channels that will be used in the future.

There is a detailed illustration of each step in C++ code in [MSCCL++ Overview](./mscclpp-overview.md). Here we only fill in some details that are glossed over there. This note only focus on the first two steps. We leave the rest of the content to the next note.

1. **Connection setup:** We recommend the programmer to store all the created connections. This is because although the communicator will record the created connections, there is no direct way to retrieve it via MSCCL++ APIs. The recommended data structure is `std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>`, which maps a rank (other than itself) to a connection.

   **A connection is bidirectional and is between two processes.** Connection setup (`Communicator::connectOnSetup()`) is designed to be async (i.e., returning immediately to the host code). Its actual call is postponed until `Communicator::setup()`. MSCCL++'s `NonblockingFuture<>` wraps around `std::future<>`, the standard C++ template class C++ for asynchronous operations. After `setup()` returns, we fetch the result of an async operation via its `get()` method.

   **Be careful with the transport in the connection setup, which is the last parameter in `Communicator::connectOnSetup()`.** Three are available: `Transport::CudaIpc` (intra-node only), `Transport::IB[n]`, and `Transport::Ethernet`. See the Communicator Implementation section below for a detailed explanation.

2. **Buffer registration: We recommend allocating GPU memory via `mscclpp::allocSharedCuda<>()`, which wraps around `cudaMalloc()`.** The additional benefit is that the returned pointer is a smart pointer, whose deleter takes cares of freeing this GPU buffer via `cudaFree()` when its last reference is destructed.

   For each buffer, we call `RegisterMemoryPairs()` to register it with the communicator using `Communicator::registerMemory()`. The returned `RegisteredMemory` points to local memory **(i.e., it does not own the memory)** and contains the information that will be sent to other ranks for them to access this piece of memory. Then, for each buffer, we exchange the memory with each peer using **(two-sided and sync)** `Communicator::sendMemoryOnSetup()` and `Communicator::recvMemoryOnSetup()`. `Communicator::sendMemoryOnSetup()` sends the information required for other ranks to access the local buffer (from their perspective, it is remote). `Communicator::recvMemoryOnSetup()` generates a new `RegisteredMemory` that holds all the information required to access the remote buffer. Addresses to this buffer will be translated to memory owned by another process.

## Communicator Implementation

**Connection setup**

- The constructor does nothing but records the bootstrap object.

- **Communicator::connectOnSetup(remoteRank, tag, localConfig)** internally creates a `Connector` that represents an async operation. During creation, `Connector` creates a CQ and QP for IB transport or sets up a listening socket for Ethernet transport. The created `Connector` is pushed into a queue internal to the communicator. This queue stores async objects and is flushed upon `Communicator::setup()`. These async objects are called setupable objects in MSCCL++, because they have a `beginSetup()` and `endSetup()` method that will be executed during `Communicator::setup()`.

  The last argument, `localConfig` of type `EndpointConfig`, holds the transport method (either IPC for intra-node comm, and IB/Ethernet for inter-node comm), as well as the default IB parameters (e.g., CQ size). Therefore, it is important to specify the correct transport method for a connection. Note that the transport of IB is provided as eight predefined enum values, ranging from `Transport::IB0` to `Transport::IB7`. The meaning of `Transport::IBn` is the `n`-th IB card on the machine, or the `n`-th IB device returned by `ibv_get_device_list()`, so the MR/CQ/PD will be registered per a specific RNIC.

- **Communicator::setup()** flushes all the setupable objects in an internal queue by calling their `beginSetup()` and `endSetup()` methods. Take `Connector`, for example.

  - Its `beginSetup()` sends the address of the local end to the remote end via the bootstrap network. This address includes the host hash, and the QP info for IB transport or the socket address for socket transport.
  - Its `endSetup()` receives the address and connects to the remote end. Once the connection is established, it is stored in an internal hash table, where the key is the shared pointer to the `Connection` object and the value is the `(remoteRank, tag)` pair of the peer. Note that `Connection` is an abstract class, whose `write()` and `flush()` methods are pure virtual functions. The created connection depends on the transport, which can be one of the three derived classes, namely `CudaIpcConnection`, `IBConnection`, or `EthernetConnection`. They implement the concrete methods. Take `IBConnection`, for example. During `endSetup()`, it transitions the QP state to RTR and RTS sequentially. For `cudaIpcConnection`, it checks that the two ends are indeed on the same machine using the host hash.

**Buffer registration**

- **Communicator::registerMemory()** returns a `RegisteredMemory` representing a block of local memory registered to the communicator. Based on the possible transport methods used with this buffer, there are different ways of registration. Note that you should call this method once for every buffer, specifying all transports at once by ORing all the transport flags together.

  - If `cudaIpc` is included, the registration uses `cuMemExportToShareableHandle` for a buffer allocated with cuMem APIs or `cudaIpcGetMemHandle` otherwise. In either case, a shareable handle will be created, which can be sent to the other process for them to retrieve this GPU memory (remote from other ranks' perspective).
  - For each IB device included in the transport, the registration calls `ibv_reg_mr()` internally. Each IB device has its own PD, and the buffer will be registered in each device's PD. Of course, the same memory buffer can be registered multiple times (even with different permissions), and each registration results in a distinct set of keys. Like shareable handles above, the keys will be provided to the remote peers to expose this local buffer (remote from other ranks' perspective). Since multiple devices can access the same buffer, it is the programmer's responsibility to avoid data hazards.

  > **Note:** Although a memory buffer can be accessed via multiple transports, each connection created by `Communicator::connectSetup()` supports only one transport.

- **Communicator::sendMemoryOnSetup()** creates another setupable object `MemorySender` and pushes it into the communicator's setup queue. Its `beginSetup()` method encapsulates all shareable handles and RDMA keys of a buffer in a message and sends it to another peer via the bootstrap network. Its `endSetup()` method is empty.

- **Communicator::recvMemoryOnSetup()** creates another setupable object `MemoryReceiver` and pushes it into the communicator's setup queue. Its `beginSetup()` method is empty. Its `endSetup()` method receives the above message and creates a new `RegisteredMemory` accordingly, which represents the remote memory. Based on how the remote memory is registered, it takes different workflows to obtain a usable local pointer.

  - For a local buffer received (i.e., host hash and pid hash both match, on the same GPU), it simply records the received data pointer.
  - For a memory allocated with cuMem API, it imports the shareable handle by `cuMemImportFromShareableHandle()`, reserves a virtual address (VA) range via `cuMemAddressReserve()`, and maps the returned handle to this VA range via `cuMemMap()`. Accessing addresses in this range will translate to accessing the remote buffer.
  - For a memory allocated with CUDA IPC API, it calls `cudaIpcOpenMemHandle()`. This function opens an interprocess memory handle exported from another process. It returns a device pointer usable in the local process. Ever since, accessing addresses in this range will translate to accessing the remote buffer.
  - For IB transport, it just records the provided RDMA keys in the `RegisteredMemory`. Ever since, it can access the remote buffer via a desired IB device by providing the corresponding keys that belong to that device's PD. In this case, it sets the remote buffer's address to 0 as a placeholder.

  > **Note:** If you print the address of the remote `RegisteredMemory`, it is usually different from the address in the remote process. This is because the printed address is the VA allocated in the receiving process, which has nothing to do with the VA allocated in the sending process.