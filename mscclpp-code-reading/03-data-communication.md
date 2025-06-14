# MSCCL++ Deep Dive 03: Data Communication

We continue our study in [02-communicator](02-communicator.md). There are two ways to perform data communication, depending on whether the host- or device-side interfaces are used.

We first introduce the host-side interfaces, because they only rely on objects we introduced in [02-communicator](02-communicator.md). Next, we will introduce MSCCL++ objects needed by device-side interfaces, namely semaphore and proxy service. Finally, we will investigate device-side interfaces.

## Host-side Data Communication

Recall we built a connection via `Communicator::connectOnSetup()`. The result of this async operation is a shared pointer to a `Connection` object. This class is an abstract class, whose `write()` and `flush()` methods are pure virtual functions. Based on the specified transport in `Communicator::connOnSetup()`, the created connection is either a `CudaIpcConnection`, an `IBConnection`, or an `EthernetConnection`.

> **Note:** Unlike connection setup or buffer registration, the `write()` method is one-sided in all derived classes. That is, the receiver does not have to post any requests. We explain below how each derived class achieves this property.

- **CudaIpcConnection**

  - **CudaIpcConnection::write(dst, dstOffset, src, srcOffset, size)** calls `cudaMemcpyAsync()` to transfer data. Each connection has its own stream that does not synchronize with the default stream. Thus, different connections launch `cudaMemcpyAsync()` to different streams. Note that `cudaMemcpyAsync()` itself is a one-sided operation.
  - **CudaIpcConnection::flush(timeoutUsec)** calls `cudaStreamSynchronize()` to wait for all previous work items in the stream to finish.

- **IBConnection**

  - **IBConnection::write(dst, dstOffset, src, srcOffset, size)** creates a WR containing the corresponding key fetched from `RegisteredMemory` (i.e., the `dst` argument). It then posts this WR using `ibv_post_send` with opcode `IBV_WR_RDMA_WRITE`, which is a one-sided operation; therefore, the receiver is not involved. It increments an internal counter counting the posted items so far.
  - **IBConnection::flush(timeoutUsec)** polls for the CQE for all posted items and checks their status. It throws an exception if any CQE status is not successful.

- **EthernetConnection**

  - **Initialization:** It connects to the remote peer and starts a background thread that receives from the socket.  It also creates a send and receive buffer in host memory upon initialization. Since `Communicator::connectSetup()` is two-sided, we will create a pair of sockets in each process, one for sending messages to the other and another for receiving.
  - **EthernetConnection::write(dst, dstOffset, src, srcOffset, size)** first copies GPU memory to the send buffer and then sends it via the socket send method. The background thread spawned on the remote peer will receive the data.
  - **EthernetConnection::flush(timeoutUsec)** does nothing, since `EthernetConnection::write()` returns when all the data have been sent.

  > **Note:** The socket send()/recv() functions are inherently two-sided. To create a one-sided `write()` interface, it spawns a background thread that continually tries to receive data from the socket. Upon receiving any data, this thread copies them from the host-side receive buffer to the GPU buffer using `mscclpp::memcpyCuda<T>()`.

## Semaphores

A semaphore is a synchronization mechanism that allows the local peer to wait for the remote peer to complete a data transfer. It contains three `std::unique_ptr<uint64_t>`s: an inbound value, an expected inbound value, and an outbound value.

- **The inbound value** is incremented by the remote peer and is waited on by the local peer.
- **The expected inbound value** is incremented by the local peer and is compared to the inbound value.
- **The outbound value** is incremented by the local peer and copied to the remote peer's inbound value.

Each semaphore has a `signal()` method and a `wait()` method. The `signal()` method increments the outbound value and sends the updated value to the remote peer's inbound value. The `wait()` method increments the expected inbound value and spins until the inbound value catches up.

Based on the type of semaphore, the three values above lie in either host memory or GPU memory, and the two methods are either host functions or device functions. There are three types of semaphores in MSCCLPP.

- **Host2HostSemaphore** sends signals from the local host to the remote host. Thus, all three values are on the host. `wait()` and `signal()` are host functions.
- **Host2DeviceSemaphore** sends signals from the local host to the remote device. Thus,  the two inbound values are on the GPU, and the outbound value is on the host. `wait()` is a device function, and `signal()` is a host function.
- **SmDevice2DeviceSemaphore** sends signals from the local device to a peer device. Thus, all three values are on the device. `wait()` and `signal()` are device functions.

Each semaphore is associated with a connection, since it needs a connection to expose its inbound value to a remote process. Moreover, `Communicator::setup()` is required after creating semaphores to let them exchange memories that store the inbound value. For performance reasons, we will mainly use `Host2DeviceSemaphore` and `SmDevice2DeviceSemaphore`. Let's take a closer look.

- **Host2DeviceSemaphore**
  - All three values are initialized as 0 (true for all kinds of semaphores).
  - **signal()** invokes the corresponding connection's `updateAndSync()` function. This function directly increments the outbound value since it's in the host memory and the function is executed by CPU. It then copies the new value to the remote peer using the transport associated with the connection (the same process as buffer registration).
  - **wait()** is a device function, so it is not a method of `Host2DeviceSemaphore` but rather an inner class `Host2DeviceSemaphore::DeviceHandle`. This class contains the two pointers to GPU-resided data plus a `wait()` method. Under the hood, it uses CUDA's `atomic_ref<>::load()` method to load the data atomically. Only 1 GPU thread is required to call the `wait()` method.
  - **Fence:** `signal()` ensures all previous work items are done when `wait()` returns, since work items in a connection are executed in order (although async to the posting thread). That is, different `cudaMemcpyAsync()` calls (in the stream associated with a connection) or different `ibv_post_send` requests are executed in their issueing order.
- **SmDevice2DeviceSemaphore**
  - It works for GPUs on the same host machine.
  - Similarly, `wait()` and `signal()` are methods of the inner class `SmDevice2DeviceSemaphore::DeviceHandle`. Only 1 GPU thread is required to call both functions. `signal()` increments the outbound value, and uses CUDA's `atomic_ref<>::store()` to store the new value to the remote device's inbound semaphore. `wait()`  uses `atomic_ref<>::load()` to load the inbound value and compare it with the expected value.
  - **Fence:** `signal()` ensures that all previous work items are completed when wait() returns, as the memory order specified in `atomic_ref<>::store()` or `load()` methods guarantees this (akin to a memory fence on a CPU).

## Proxy Service

For communication across the network, the current computer architecture requires the CPU to post RDMA requests. Similar to NCCL, MSCCLPP achieves this through a GPU-CPU FIFO mechanism, where GPU threads post requests to a FIFO queue (residing in unified memory, accessible by both GPU and CPU), and a CPU proxy thread fetches the requests. Based on the request type, the proxy thread either initiates RDMA transactions or polls for completion status.

- **Fifo** is the FIFO queue in MSCCLPP. This queue should support multiple producers (each is a GPU thread) and a single consumer (i.e., the proxy thread). Accordingly, `push()` is a device interface that belongs to an inner class, `FifoDeviceHandle`, and `pop()` is a host interface. The queue has a head and tail count, both of which start from 0 and only increase. Each request in the queue is of type `ProxyTrigger`. Below, we list the states of a `Fifo` and its `FifoDeviceHandle`.

  - **Fifo states:**

    1. Pointer to the queue of `ProxyTrigger`s, allocated on host via `cudaHostAlloc()` (as the pinned memory, and GPU can access it)

    2. A `hostTail` count, allocated on host, only accessed by the host. However, the GPU needs the `hostTail` to ensure that the queue is not full when pushing new requests. Therefore, CPU needs to flush its tail to GPU from time to time.

       > **Note:** GPU actually has another way of doing this by checking whether the next slot is non-empty. This can eliminate a replica of `hostTail` on GPU. However, this is inefficient as every time GPU has to access CPU memory. Therefore, MSCCLPP takes the method of maintaining a replica of `hostTail` on GPU.

    3. The queue `size` (a constant)

    4. A dedicated CUDA `stream` for flushing `hostTail` to GPU (a constant)

  - **FifoDeviceHandle states:**

    1. Pointer to the queue, which is copied from the host-side pointer
    2. `tailReplica`, allocated on GPU, which traces the value of `hostTail`. Note that the property `tailReplica <= hostTail` always holds.
    3. A `head` count, allocated on GPU, only accessed by the device. CPU does not need this state because it only reads from the queue tail. It pauses on seeing an empty `ProxyTrigger` and resumes when the tail request is non-empty.

  Now the implementation of their interfaces should be fairly easy to understand.

  - **push()** first increments the `head` atomically (w.r.t. each calling GPU thread). If the `head` is no more than `size + railReplica`, it then writes to the current slot.
  - **poll()** (a CPU interface) loads the trigger at index `hostTail`
  - **pop()** resets the tail slot and increase `hostTail` (non-atomic since there is only one consumer)
  - **flushTail()** (a CPU interface) uses `cudaMemcpyAsync()` to copy `hostTail` to `tailReplica` in a dedicated `stream`.
  - **sync()** waits until `head == tailReplica` and the corresponding slot is empty.

- **ProxyHandler** holds the processing logic of all kinds of `ProxyTrigger`. It is defined as a function pointer whose input is a `ProxyTrigger` and whose return value is a `ProxyHandleResult`. Although `ProxyTrigger` is defined as two `uint64_t`, when parsing the type and arguments of a request, we convert it to `ChannelTrigger` first. Currently, there are three types of requests: data transfer, signaling, and flushing. On the other hand, `ProxyHandleResult` dictates the state transition of the proxy thread, which has three values: `continue`, `FlushFifoTailAndContinue`, and `Stop`.

  The default `ProxyHandler` works as follows.

  - For a data transfer request, it fetches the src/dst `RegisteredMemory` plus the offset and posts the requests using the corresponding connection's `write()` method. Recall that `write()` is one-sided.
  - For a signaling request, it uses the corresponding semaphore's `signal()` method. Note that only `Host2DeviceSemaphore` can be used.
  - For a flushing request, it uses the corresponding connection's `flush()` method. The return value is `FlushFifoTailAndContinue`.

- **Proxy** implements the proxy thread logic. It holds a `Fifo` queue, uses a `ProxyHandler` to process all the requests, and supports a `threadInit` method that is first executed upon the start of the proxy thread. `Proxy::start()` spawns the proxy thread, and `Proxy::stop()` terminates and joins the proxy thread.

  `Proxy` implements a default proxy thread logic. After calling `threadInit`, it polls the `Fifo` queue in an infinite loop. Upon seeing a non-empty request, it invokes `ProxyHandler` to process the request. Done processing, it invokes `Fifo::pop()` to pop the queue. Moreover, every `ProxyStopCheckPeriod` loops it checks for the stop signal, and every `flushPeriod` loops it flushes the `Fifo`'s `hostTail` to GPU. So no matter whether the user calls `flush()`, GPU's `tailReplica` is not too stale.

  There is hardly any need of customizing `Proxy`. To let MSCCLPP process new types of requests, it suffices to customize `ProxyHandler` that implements the request handling logic.

- **ProxyService** implements the default version of proxy services. Its constructor takes a Fifo size argument, and constructs a `Proxy` whose `ProxyHandler` is the default one and `threadInit` binds the proxy thread to the NUMA domain of the GPU. We use `ProxyService::startProxy()` to start the proxy thread (which calls `Proxy::start()` internally) and `ProxyService::stopProxy()` to end it (which calls `Proxy::stop()` internally). However, we must do two things ahead of `ProxyService::startProxy()`.

  1. Build semaphores using `ProxyService::buildAndAddSemaphore(communicator, connection)`, since signaling and flushing requests are operations of semaphores. This method constructs a `Host2DeviceSemaphore` and returns a semaphore ID. 

  2. Build a proxy channel using `ProxyService::proxyChannel(sid, dst, src)`, since a GPU-to-GPU channel is associated with a semaphore and the src/dst memory buffer. We will introduce `proxyChannel` in the next section.

     > **Note 1:** This document is based on MSCCLPP v0.6.0, the latest version at the time of writing. Since then, `ProxyChannel` is renamed to `PortChannel`, and is partially refactored to be consistent with the MSCCLPP paper.

     > **Note 2:** `ProxyService` is just a helper class that builds all the semaphores and `ProxyChannel`s. In a typical MSCCLPP program, one `ProxyService` suffices.

     > **<font color="red">CAUTION:</font>** If you defined `ProxyService` or `Communicator` as  a global variable, remember to manually delete it before `main()` returns. This is because during destruction, some of its member still makes CUDA calls. If any of them is global and is not manually deleted, its destruction can occur after the CUDA driver is shut down. Similar suggestions are given for other MSCCLPP objects.

## Device-side Data Communication

> **Note:** This document is based on MSCCLPP v0.6.0, the latest version at the time of writing. Since then, the channel name and structure is changed. Yet, the key idea remains the same.

Depending on the underlying connection, MSCCLPP provides two types of channels: `SmChannel` for intra-node communication and `ProxyChannel` for inter-node communication.

- **ProxyChannel:** We recommend building `ProxyChannel` with `ProxyService::proxyChannel(sid, dst, src)`, where `sid` is the semaphore ID returned by `ProxyService::buildAndAddSemaphore(communictor, connection)`. It represents communication between GPU-to-GPU via the assistance of the proxy thread, so all its interfaces are device-side (or interfaces of an inner class `ProxyChannelDeviceHandle`).

  - Sender-side

    - **put(dstOffset, srcOffset, size):** Only one GPU thread is required to call this function. It constructs a `ChannelTrigger` and uses `Fifo::push()` to post a data transfer request to CPU.
    - **signal():** Only one GPU thread is required to call this function. It constructs a `ChannelTrigger` and uses `Fifo::push()` to post a signaling request to CPU.
    - **flush():** Only one GPU thread is required to call this function. It constructs a `ChannelTrigger` and uses `Fifo::push()` to post a flushing request to CPU. It then uses `Fifo::sync()` to wait until the Fifo queue is empty.
    - **putWithSignal(dstOffset, srcOffset, size):** Rather than posting two requests, this method only posts one request, whose request type is the OR of the two.
    - **putWithSignalAndFlush(dstOffset, srcOffset, size):** Rather than posting three requests, this method only posts one request, whose request type is the OR of the three.

    > **Note:** You don't specify src and dst buffers because they are already specified in ``ProxyService::proxyChannel`.

  - Receiver-side

    - **wait():** It directly calls semaphore's `wait()` method to wait until all the tasks before the sender's corresponding `signal()` are finished.

    > **Note:** There is a one-to-one match between sender's `signal()` and receiver's `wait()`. `wait()` increases the expected inbound value, and waits for that inbound value to arrive. `signal()` increases the outbound value and sends it to update the receiver's inbound value.

- **SmChannel:** Since there is no helper class, we need to construct a `SmDevice2DeviceSemaphore` from sratch and then builds the `SmChannel`. Its construction needs `(semaphore, dst, src, Packet buffer)`, where the last one is a buffer for LL (low-latency) protocols. We will explain it in more details below.

  - **SmDevice2DeviceSemaphore::SmDevice2DeviceSemaphore(communicator, connection):** When used upon CudaIpc transport, it sets up a local `uint64_t` on GPU memory and exchanges it with the remote peer  using the buffer registration mechanism above. Therefore, a `Communicator::setup()` is required after this function.

  - **SmChannelDeviceHandle:** All methods are device interfaces.

    - **put<Alignment, CopyReminder>(dstOffset, srcOffset, numOfBytes, threadId, numThreads):** This function is intended to be collectively called by multiple threads. Each thread copies a part of data. It implements the HB protocol (High-bandwidth, high-latency due to a sync after the call).

      - `Alignment` must be 4, 8, or 16
      - `CopyReminder` is a boolean flag dictating whether to copy the misaligned bytes (if there are any at the front or the rear)
      - `numOfBytes` is the total number of bytes to be copied by the calling threads. It must be a multiple of `Alignment`.
      - `threadId` is the index of the current thread among all threads running this function. This is different from the `threadIdx` in CUDA.
      - `numThreads` is the number of threads running this function.

      Its implementation is as follows. There is an inner method `copy<T>(T *dst, T *src, numElems, threadId, numThreads)`, where each thread copies a value of type `T` from `src` to `dst`. `threadId` determines the starting index of a GPU thread. This index is increased by `numThreads` each time (i.e., pointing to the `T` element `numThreads` later). The loop stops when the pointer reaches or surpasses `numElems`. 

      Now we go back to its implementation. If `CopyReminder==true`, then the misaligned bytes at the beginning are copied with `copy<int>`. Then, the aligned bytes are copied by `copy<int/long long/longlong2>` if `Alignment==4/8/16`, respectively. Finally, the misaligned bytes at the rear are copied with `copy<int>`.

      > **Note:** Restricted by the current MSCCLPP implementation, you should always make sure any byte offset and size passed to these functions are a multiple of 4 at least. Although in practice, there is seldom usage of an offset or a size not divisible by 4.

    - **get<Alignment, CopyReminder>(remoteOffset, localOffset, numOfBytes, threadId, numThreads)** works similarly as `put`. To accomplish a transaction, either the sender calls `put` or the receiver calls `get`.

    - **putPackets<PacketType>(dstOffset, srcOffset, numOfBytes, threadId, numThreads, flag):** It supports two LL protocols: LL8 and LL16. It constructs the corresponding LLPacket from the data in the local memory (src) and write it on the *remote packet buffer* (dst, which is the last argument in `smChannel`'s constructor). This function is intended to be collectively called by multiple threads. Each thread copies a part of data. It implements the LL protocol (low-bandwidth, low-latency due to atomic writing of data with flags).

      - `PacketType` must be `LL16Packet` (8-byte data+8-byte flag) or `LL8Packet` (4-byte data+4-byte flag)
      - `numOfBytes` is the total amount of bytes to be copied by all the participating threads, which must be a multiple of 4/8 for `LL8Packet`/`LL16Packet`
      - `threadId` and `numThreads` are the same as above
      - `flag` is the 32-bit flag value to write

      It just wraps around two inner methods: `putLL8Packets()` and `putLL16Packets()`. 

      - **putLL8Packets():** Each `LL8Packet`'s first 4 bytes are the data, and the last 4 bytes are the flag. Each participating thread starts from the `threadId`-th 4-byte word since the `srcOffset`, and writes to the 8-byte `LL8Packet` at the same index using a special PTX code `st.volatile.global.v2.u32`. It increases the index by `numThreads` in the next loop, and returns when the index reaches `numOfBytes/4`.
      - **putLL16Packets():** Each `LL16Packet`'s layout is 4-byte data1 + 4-byte flag1 + 4-byte data2 + 4-byte flag2. Each participating thread starts from the `threadId`-th 8-byte word since the `srcOffset`, and writes to the 16-byte `LL16Packet` at the same index using a special PTX code `st.volatile.global.v4.u32`. It increases the index by `numThreads` in the next loop, and returns when the index reaches `numOfBytes/8`.

    - **getPackets<PacketType>(targetOffset, srcOffset, numOfTypes, threadId, numThreads, flag)** works cooperatively with `putPackets<>()`. It extracts the data from the local packet buffer (`target`) and writes them to the local buffer (`src`).

      It just wraps around two inner methods: `getLL8Packets()` and `getLL16Packets()`.

      - **getLL8Packets():** Each participating thread starts from the `threadId`-th `LL8Packet` in the packet buffer since the `targetOffset`, and writes to the 4-byte word in the src buffer at the same index. Each write is done via repetitively loading a `LL8Packet` via a special PTX code `ld.volatile.global.v2.u32` and checking its flag field matches the `flag` argument. Then, it increases the index by `numThreads` in the next loop, and returns when the index reaches `numOfBytes/4`.
      - **getLL16Packets():** Each participating thread starts from the `threadId`-th `LL16Packet` in the packet buffer since the `targetOffset`, and writes to the 8-byte word in the src buffer at the same index. Each write is done via repetitively loading a `LL16Packet` via a special PTX code `ld.volatile.global.v4.u32` and checking its two 4-byte flag fields both match the `flag` argument. Then, it increases the index by `numThreads` in the next loop, and returns when the index reaches `numOfBytes/8`.

      > **Note:** While `put` and `get` are one-sided opeartions, `putPackets` and `getPackets` are two-sidded, i.e., the sender calls `putPackets` and the receiver calls `getPackets` to accomplish a transaction. This is because `putPackets` does not directly write to the receiver's destination buffer, but its packet buffer. `getPackets` extracts the data from its local packet buffer into the destination buffer.
