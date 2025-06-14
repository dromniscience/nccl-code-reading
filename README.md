# NCCL & MSCCL++

This repo contains two series of notes: one focuses on [NCCL v2.25.1](https://github.com/NVIDIA/nccl/tree/v2.25.1-1), and the other focuses on [MSCCL++ v0.6.0](https://github.com/microsoft/mscclpp/tree/v0.6.0). Each series begins with an overview note, followed by subsequent notes that explain one of the major components in the corresponding project.

We recommend that readers have an overall understanding of NCCL first, as MSCCL++ borrows many mechanisms from NCCL. One of the major issues with NCCL is that its code is unarguably cumbersome and its components are tightly coupled, which makes it hard to customize or extend with aggressive features.

On the other hand, MSCCL++ is a nice redesign of the core of NCCL, which decouples basic components as much as possible. Although it does not reimplement everything in NCCL, MSCCL++ provides us with a controllable and understandable framework. We can utilize its primitives and abstractions to develop a customized GPU-oriented collective communication library from scratch without reinventing the wheel. To illustrate, we give a clean-slate implementation of an All-Pairs All-Reduce GPU kernel in the [MSCCL++ overview](./mscclpp-code-reading/mscclpp-overview.md).

Hope you have an enjoyable read with these notes!