# MSCCL++ Deep Dive 01: Bootstrap

## Bootstrap Creation

There are two ways to create a bootstrap network, distinguished by how the boostrap socket address is chosen. 

1. Let the root rank search this address using the filter provided as an env var `MSCCLPP_SOCKET_IFNAME`. We create the bootstrap network per process in MSCCLPP via the following code.

   ```C++
   std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, totalRanks);
   {
         mscclpp::UniqueId id;
         if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId(); // Set MSCCLPP_SOCKET_IFNAME for choosing the bootstrap interface
         MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
         bootstrap->initialize(id);
   }
   bootstrap->initialize(id);
   ```

   Same as NCCL, this unique ID stores the common address of the bootstrap socket for others to connect.

2. Users specify an explicit address (e.g., `10.10.10.10:12345`). They can utilize the following snippet.

   ```
   std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, totalRanks);
   const std::string IfPort = "10.10.10.10:12345";
   bootstrap->initialize(IfPort); // 2nd version of initialize(), which accepts a std::string
   ```

Either way, we obtain a functional bootstrap network among all processes. We can invoke methods or operations of the bootstrap network introduced below.

## Bootstrap Operations

Bootstrap network supports these operations:

1. A global barrier via `TcpBootstrap::barrier()`
2. All-Gather via `TcpBootstrap::AllGather()`
3. P2P send/receive via `TcpBootstrap::send()` and `TcpBootstrap::recv()`. **P2P send/receive is two-sided (i.e., sender must call `send()` and receiver must call `recv()` to accomplish a transaction).** One argument in P2P send/recv is a message tag. This `tag` serves the same purpose as in MPI: differentiating multiple connections between the same pair of ranks.

## Bootstrap Implementation

- **TcpBootstrap::createUniqueId()** uses a bootstrap interface selection procedure similar to NCCL. Implemented in `FindInterfaces()`, it finds a suitable interface based on the env var `MSCCLPP_SOCKET_IFNAME`.

- **TcpBootstrap::initialize()** uses a similar procedure to NCCL. There are two versions of `TcpBootstrap::initialize()`: One accepts a unique ID, and the other accepts an IP:port string. Both versions work alike as follows:
  
  1. Rank 0 creates another thread whose logic is implemented in `bootstrapRoot()`. This thread (1) waits for all ranks to connect, and (2) sends the address of Rank $(i+1)$ to Rank $i$.
  2. Each rank uses the `FindInterfaces()` to find an available address and creates a listening socket in the bootstrap network. It then sends this address to the above thread in Rank 0 and receives the next peer's listening address. Next, it connects to the next peer and accepts the connection from the previous peer. At this point, a ring bootstrap network is established.
  3. Each rank finally calls `TcpBootstrap::allGather()` so that every rank knows the listening addresses of the other ranks, which are stored in the bootstrap object.
  
- **TcpBootstrap::barrier()** is implemented as a `TcpBootstrap::AllGather()`.

- **TcpBootstrap::AllGather()** implements a Ring All-Gather, same as NCCL.

- **TcpBootstrap::send()** and **TcpBootstrap::recv()** wrap around the socket send and recv operations. The concrete socket to use is determined by `(peer, tag)`. Therefore, messages of different tags (even to the same peer) go through different sockets. If a `(peer, tag)` combination is first seen, the sender and receiver create a new socket.

  > **Note:** For correct programs, the sender and receiver will both regard the tuple `(peer, tag)` as the first seen.