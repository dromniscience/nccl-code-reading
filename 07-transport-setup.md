# NCCL Communicator Initialization #: Transport Setup 

## Control Flow

[ncclCommInitRank()](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1715)

- [`ncclCommInitRankFunc()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1340)
  - [`initTransportsRank()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L659) <- We are here!

In the previous note, we learned how channels on different machines are connected. Note that all the previous steps in channel search are just logical: It computes which peers I should connect to, the number of channels I have, and the path types between me and the peers. This step will show how NCCL turns this logical description into physical connections.

## Transport Setup: Preparation

Recall that [`ncclTopoPostSet()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1026) connects channels on participating machines in rings, trees, and collNets. The control flow then goes to [`ncclTopoComputeP2pChannels()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1046), which determines `comm->p2pnChannels` (number of channels needed by P2P functions) and `comm->p2pnChannelsPerPeer` (the number of channels used by every P2P connection). Both of them are [a power of two](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/paths.cc#L802-803).

Now it's time to initialize channels via [`initChannel()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/channel.cc#L12), which are stored in [`struct ncclChannel`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/comm.h#L139). `initChannel()` allocates the arrays in `ncclChannel`, and it is called by both [`ncclTopoComputeP2pChannels()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1046) and [`setupChannel()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1143). One of the key members of `ncclChannel` is an array of [`ncclChannelPeer`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/comm.h#L140), each containing several send and receive [`ncclConnector`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/device.h#L199-203)s. We highlight the meanings of its members below. Note that [`ncclTransportComm`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/transport.h#L96) is an abstraction of any transport method if it implements the required functions like `setup()` and `connect()` in this class. NCCL prefines [four means of transport](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/transport.h#L25-L29): peer access (`p2p`), shared memory (`shm`), network (`net`), and collective network (`collNet`). Each transport has a [send](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/transport.h#L112) `ncclTranpsportComm`, and a [receive](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/transport.h#L113) `ncclTranpsportComm`. Note that the `net` transport itself is again an abstraction of the actual network type, which can be a user plugin, RDMA, or socket network.

```C++
// [device.h](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/device.h#L134)
struct ncclConnector {
  int connected; // state of this connection, established?
  struct ncclProxyConnector proxyConn; // connection to the proxy thread
  struct ncclTransportComm* transportComm; // abstraction of transport
  void* transportResources; // buffer in this connection
  struct ncclConnInfo conn; // other states of the connection
};

// [device.h](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/device.h#L104)
struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};
```

## Transport Setup

`initTransportsRank()` then calls [`ncclTransportRingConnect()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1145), [`ncclTransportTreeConnect()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1148), [`ncclTranportsPatConnect()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1151), etc. to realize the channels. Under the hood, they all call [`ncclTransportP2pSetup()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport.cc#L101) to do the real stuff. It [tries](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport.cc#L27) the four means of transport between me and the peer, [finds](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport.cc#L31) the first accessible transport, and [sets](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport.cc#L34) it up with the corresponding transport. The trying order is [p2p>shm>net>collNet](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport.cc#L14-L19). We emphasize that this function is done per send/receive peer per channel. For example, if Rank 0 has two channels, where it sends to Rank 1 and receives from Rank 15 in Channel 0 and 1, then Rank 0 will connect to Ranks 1 and 15 twice.

We use two examples to illustrate the connection setup: the RDMA network transport and p2p transport.

**Case study #1: RDMA setup.** [`netTranport`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net.cc#L1718)'s methods (e.g., `sendSetup()`, `sendConnect()`) rely on the proxy thread to build the real network connection. It pushes the [request](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/proxy.h#L368-L381) to a queue owned by the proxy thread, and this thread [fetches and processes](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/proxy.cc#L1425) the request asynchronously. The actual network is stored in [`ncclProxyState`](https://github.com/NVIDIA/nccl/blob/master/src/include/proxy.h#L303), and its value is [`ncclNetIb`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L2344) for the RDMA network. Therefore, the transport is set up using the IB interfaces defined in the file [`net_ib.cc`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc). NCCL sets up the RDMA network in a standard way:

1. [Creates](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1240) a PD (Protect Domain) and CQ (Completion Queue)
2. [Creates](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1253) a QP (queue pair) of type RC (Reliable Connection) and [sets](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1081) it to INIT state
3. [Registers](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1278) a FIFO queue in an MR (Memory region) in the PD
4. [Uses](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1328) the socket network to send my local QP's information and receive remote QP's information
5. [Establishes](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1376) a connection with the remote device and sets the QP to RTR (Ready To Receive) state
6. [Sets](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/net_ib.cc#L1133) the transmission parameters of the QP and sets the QP to RTS (Ready To Send) state

**Case study #2: P2P setup.** [`p2pTransport`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L1115)'s methods also rely on the proxy thread to build the real P2P connection. If the same process manages the two ends, it uses [direct](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L451) GPU communication. Otherwise, it uses [CUMEM](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L455) APIs and resorts to legacy CUDA [IPC](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L460) APIs if CUMEM is not supported. For example, NCCL performs P2P communications with CUMEM APIs, including [`cuMemMap()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/alloc.h#L232), [`cuMemImportFromShareableHandle()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L277), and [`cuMemImportFromShareableHandle()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/transport/p2p.cc#L2277). If you are interested in CUMEM APIs, check [this doc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html).