# NCCL Communicator Initialization #08: Tuning Model

## Control Flow

[ncclCommInitRank()](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1715)

- [`ncclCommInitRankFunc()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1340)
  - [`initTransportsRank()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L659) <- We are here!

In the previous note, we learned how NCCL connects adjacent peers over several interconnections, e.g., PCIe, NVLink, and the network. The connection establishment requires the assistance of the proxy thread. We will see how NCCL executes the P2P and collective functions using these connections. But before we move on to the NCCL runtime, there is one last piece in the communicator initialization: the tuning model.

**The tuning model helps the NCCL runtime to choose the best kernel launch configuration (e.g., threads per block and the number of blocks/channels), protocol (e.g., SIMPLE/LL/LL128), and algorithm (e.g., RING/TREE, but only for collective function).** When you issue any collective (e.g., `ncclAllReduce`), the runtime calls an internal selector that walks over all candidate (algorithm, protocol) pairs and picks the one with the lowest estimated time. A similar procedure applies to P2P functions, except that no algorithm is required.

## Tuning Model: Big Picture

**[`ncclTopoTuneModel()`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc#L1214) implements the tuning model.** This is the final step in NCCL initialization: `initTransportsRank()` will return after this call. Under the hood, it fills in [four tables](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/include/comm.h#L492-L496) in `ncclComm`: `threadThresholds`, `latencies`, `bandwidths`, and `maxThreads`.

- `threadThresholds[algo][proto]`: It [decides](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L486) per-thread amount of work before NCCL increases the thread count or opens more channels; overridable with `NCCL_THREAD_THRESHOLDS`
- `latencies[func][algo][proto]`: It is the base latency (like the $\alpha$ cost in the $\alpha$-$\beta$ cost model) for each function, algorithm, and protocol combination. It includes base software latency, link latency, protocol‑specific extra flushes, and corrections for multi‑node rings/trees.
- `bandwidths[func][algo][proto]`: It is the algorithmic bandwidth (like the reciprocal of $\beta$ in the $\alpha$-$\beta$ cost model) for each function, algorithm, and protocol combination. The formula folds in `bwIntra` or `bwInter` from the topology graph, the number of ranks and nodes, architecture‑specific ceilings (`llMaxBw`, `perChMaxTreeBw`, …), and protocol overhead factors (e.g., 50% for LL, 92% for LL128). When a particular combination is disabled (e.g., due to the user-specified filter), its value is set to 0.
- `maxThreads[algo, proto]`: It sets a default number of CUDA threads to launch per block per algorithm and protocol combination; overrideable with `NCCL_NTHREADS` and `NCCL_LL128_NTHREADS`.

**When executing a collective function, the runtime estimates the completion time for a (algorithm, protocol) combination with**
$$
T_{\mathrm{predicted}}(S)=\frac{S}{\mathrm{bandwidths[func][algo][proto]}}+\mathrm{latencies[func][algo][proto]},
$$
where $S$ is the data size. It chooses the combination that achieves the minimal estimated time and launches the corresponding kernel. The launch configuration (i.e., how many threads and channels to use) comes from the `threadThresholds` and `maxThreads` values.

**On the other hand, P2P functions do not need an algorithm search, but they still use the same protocol engine and thread‑threshold logic.** We defer details to the notes on NCCL runtime.

## Tuning Model: Details

**We use `bandwidths` and `latencies` arrays as two examples to introduce how the tuning model determines their values.** Elements in both arrays are filled in a triple-nested [loop](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L232-L244) (func, algo, proto).

**Case study #1: `bandwidths`.** Based on the topology graph, the initial `busBw` for each (func, algo, proto) combination is the [product](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L252) of `graph->nChannels` and `graph->bwIntra` for single-node task or `graph->bwInter` for multi-node task. In other words, it is the ideal full‑bus bandwidth (in GB/s) the algorithm could reach if there were no software overheads. It then discounts `busBw` for protocol and hardware effects. This is done using case-by-case rules. We illustrate bandwidth discounting with five examples:

| Correction                     | Code line                                                    | Rationale                                                    |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LL halves useful bytes         | [`busBw = std::min(llMaxBw, busBw * .5);`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L255) | 4 B of every 8B is a flag in LL                              |
| Waiting time in branching tree | [`busBw *= .85;`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L260) | About 15 % off in Nvidia's microbenchmark                    |
| Only use CollNet with SIMPLE   | [`busBw = 0;`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L262-L263) | Don't use CollNet with atomic protocols                      |
| NVLS discount                  | [`ratio *= 5.0/6.0`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L292) | One sixth of the time is spent in swap stages                |
| RING algorithm                 | [`ratio *= (1.0 * nRanks) / nsteps;`](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L291) | Convert bus bandwidth to algorithmic bandwidth for the ring algorithm (the factor is $\frac{W}{2(W-1)}$) |

**Case study #2: `latencies`.** The general formula is

![eqn](https://latex.codecogs.com/svg.image?\mathrm{Lat}=\mathrm{BaseLat}+\mathrm{\&hash;IntraSteps}\times\mathrm{IntraLat}+\mathrm{\&hash;InterSteps}\times\mathrm{InterLat}.)

Here, base latency is the constant software overhead of launching the CUDA kernel and setting up FIFOs for that protocol. Its value is [hard-coded](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L118-L120) for each (algo, proto) combination. Intra-node latency is the empirical GPU‑to‑GPU time inside a node. Its value is also [hard-coded](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L130-L137) (e.g., NVLink ≈ 0.7 µs, PCIe ≈ 1.1 µs). Inter-node latency is the network RTT plus driver overhead. Its value is the [sum](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L301) of a [hard-coded](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/graph/tuning.cc#L139-L141) value and `graph->latencyInter`. Different algorithms have unique inter- and intra-node steps. For example, there are $2(W-1)$ inter-node steps in a multi-node All-Reduce and $0$ in a single-node All-Reduce.

