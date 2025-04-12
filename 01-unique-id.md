# NCCL Communicator Initialization #01: Unique ID

## Control Flow

- [ncclGetUniqueId(ncclUniqueId* out)](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L104) <- We are here!

Unique ID (defined as `struct {char [128];}`) is just an address of a listen socket `<ip>:<port>` on the calling process that other processes can connect to. Its goal is to initialize the **bootstrap network**. This network (using TCP) is used to exchange small control messages when creating an NCCL communicator. However, the fully functional bootstrap network can only be created when all other ranks join this network. Therefore, the root creates a **bootstrap thread** that listens on the bootstrap socket whose address is the ID. All participating processes will later connect this socket to build the bootstrap network. The bootstrap thread is short-lived.

## Generating Unique ID

- High-level procedure

  - [`initEnv()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L82) populates env vars
  - [`bootstrapNetInit()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L85) selects the bootstrap interface on the root
  - [`bootstrapGetUniqueId()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L108) initializes the bootstrap network on the root

- More details

  - [`initEnv()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/param.cc) first populates the key-value pairs in `${NCCL_CONF_FILE}` (or user's `~/.nccl.conf`), then populates key-value pairs in `/etc/nccl.conf`

  - There is an extra call to [`initGdrCopy()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L83) in [`ncclInit()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L91), which is controlled by [`${NCCL_GDRCOPY_ENABLE}`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L66) and is off by default. Please don't confuse it with GDR.

  - The last call to [initNvtxRegisteredEnums()](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L87) in [`ncclInit()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L91) enables NVTX to display human-readable op names for the captured traces (not interested)

  - [`bootstrapNetInit()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L92) selects the bootstrap interface on root, which is saved in the global variables [`bootstrapNetIfName`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L85) and [`bootstrapNetIfAddr`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L86).

    - It first calls [`ncclFindInterfaces()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L110) to select the bootstrap interface, which uses [`findInterfaces()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/socket.cc#L357) to iteratate over all available NICs and chooses one based on [`${NCCL_SOCKET_IFNAME}`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/socket.cc#L352).

      - [`findInterfaces()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/socket.cc#L123) uses `${NCCL_SOCKET_IFNAME}` to match all available interfaces ([`getifaddrs()`](https://man7.org/linux/man-pages/man3/getifaddrs.3.html)) using [`matchIfList()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/socket.cc#L158). The env doc [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname) explains the matching rules.

    - Bootstrap only needs one interface. You can see a bootstrap log message saying which interface it uses.

      ```
      sz-k8s-master:44274:44274 [0] NCCL INFO Bootstrap: Using ens255f0np0:10.10.10.108<0>
      ```

  - [`bootstrapGetUniqueId()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L399) passes the address of bootstrap interface to [`bootstrapCreateRoot()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L374). The later function calls [`ncclSocketListen()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L382) to open a listen socket, and spawns a thread whose entry point is [`bootstrapRoot()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L388).

    - [`ncclSocketListen()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/misc/socket.cc#L382) calls `bind()` and `listen()` C APIs under the hood

    - [`bootstrapRoot()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L267)'s main job is a while loop that waits for all the processes to [connect](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L294) and [receive](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L295) their handles. Once the handle is received, the established socket is [closed](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L296). The handle is defined as [`struct extInfo`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/bootstrap.cc#L234), which contains the rank of the connected process, the socket address of its listen socket for root (`listenRootAddress`), and the socket address of its listen socket in the ring (`connectInfo`). Root will send to rank $r$ its next peer's `connectInfo` in the ring (i.e., rank $r+1$) using its `listenRootAddress`. Once all peers know the `connectInfo` of its next peer, the bootstrap thread on root exits.

      - You can see a bootstrap log message saying how much time this thread spends on waiting, receiving, and sending messages if NCCL is compiled with trace enabled (`TRACE=1` when `make`).

        ```
        sz-k8s-master:44274:44302 [0] 614.197680 bootstrapRoot:358 NCCL TRACE Root timings (wait 0.569710, recv 0.042940, send 0.000000)
        sz-k8s-master:44274:44302 [0] 614.223818 bootstrapRoot:370 NCCL TRACE DONE
        ```
