# JAX Integration

!!! note
    IREE's JAX support is under active development. This page is still under
    construction.

## Overview (draft)

<!-- DRAFT -->
Some understanding of JAX is required

While other machine learning frameworks let you build graphs out of layers of
operations, JAX operates at a lower level, letting users compose programs out of
NumPy and other Python functions that are then just-in-time compiled and
executed.

Two operating modes:

* just-in-time (JIT) compilation for interactive use in Python
* ahead-of-time (AOT) compilation for deployment in other environments
<!-- DRAFT -->

## Old docs (remove/merge into new docs)

IREE offers two ways to interface with [JAX](https://github.com/google/jax)
programs:

* An API for extracting and compiling full models ahead of time (AOT) for
  execution apart from JAX. This API is being developed in the
  [iree-org/iree-jax repository](https://github.com/iree-org/iree-jax).
* A PJRT plugin that adapts IREE as a native JAX backend for online / just in
  time (JIT) use. This plugin is being developed in the
  [openxla/openxla-pjrt-plugin repository](https://github.com/openxla/openxla-pjrt-plugin).

<!-- TODO: Expand on interface differences -->
<!-- TODO: Add quickstart instructions -->
<!-- TODO: Link to samples -->

## References (while writing, add inline where relevant)

| Topic | Reference page |
| ----- | -------------- |
JAX | [JAX: Autograd and XLA](https://github.com/google/jax)
JAX JIT | [Just In Time Compilation with JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
JAX AOT | [Ahead-of-time lowering and compilation](https://jax.readthedocs.io/en/latest/aot.html)
PJRT Plugin | [OpenXLA PJRT Plugin](https://github.com/openxla/openxla-pjrt-plugin)
