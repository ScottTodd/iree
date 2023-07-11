# C API bindings

!!! note - "Planning notes"

    [Discussion thread](https://discord.com/channels/689900678990135345/689900680009482386/1123726574525612032)

    - [ ] Compiler via C API
    - [ ] Runtime via C API
    - [ ] [runtime-library](https://github.com/iree-org/iree-samples/tree/main/runtime-library)
    - [ ] [iree-template-runtime-cmake](https://github.com/benvanik/iree-template-runtime-cmake)
    - [ ] [iree-template-cpp](https://github.com/iml130/iree-template-cpp)
    - [ ] `libIREECompiler.so`
    - [ ] CMake installed development packages

    > Integrators often like to link against a single runtime library, and the method of doing so is naturally owned by the integrator -- not necessarily by IREE itself (i.e. IREE will never release a full libireert.a). This separation of concerns is of practical importance because IREE's low level runtime API is fine-grained and geared towards usage via LTO style optimizations. This makes it inherently non-distributable, since every toolchain defines such features and interop differently.
    >
    > Also, it is often convenient during early development (of language bindings, etc) to simply dynamically link to something that works, even if not optimal. Because it cannot produce the best integration, IREE itself does not export a shared runtime library. However, to aid development, it can be useful for users to produce one.

## Overview

The IREE compiler and IREE runtime both have their own C/C++ APIs for use in
other projects.

!!! note

    There are multiple ways to distribute and depend on C/C++ projects, each
    with varying levels of portability, flexibility, and toolchain
    compatibility. IREE aims to support common configurations and platforms.

### Compatibility

## Compiler API

The IREE compiler is structured as a monolithic shared object with a dynamic
plugin system allowing for extensions. The shared object exports symbols for
versioned API functions.

!!! Tip "Tip - building from source"

    When [building from source](../../building-from-source/getting-started.md),
    some components may be disabled to reduce binary size and improve build
    time. There are also options for using your own LLVM or linking in external
    target backends.

```mermaid
graph TD
  accTitle: IREE compiler linkage model diagram
  accDescr {
    The libIREECompiler.so or IREECompiler.dll shared object contains pipelines,
    target backends, and general passes as private implementation details.
    Compiler plugins interface with the compiler shared object to extend it with
    custom targets, dialects, etc.
    Applications interface with the compiler shared object through the compiler
    C API's exported symbols.
  }

  subgraph compiler[libIREECompiler.so / IREECompiler.dll]
    pipelines("Pipelines

    • Flow
    • Stream
    • etc.")

    targets("Target backends

    • llvm-cpu
    • vulkan-spirv
    • etc.")

    passes("General passes

    • Const eval
    • DCE
    • etc.")
  end

  plugins("Compiler plugins

    • Custom targets
    • Custom dialects
    • etc.")

  application(Your application)

  compiler <-- "Plugin API<br>(static or dynamic linking)" --> plugins
  compiler -. "Compiler C API<br>(exported symbols)" .-> application
```

API definitions can be found in the following locations:

| Source location | Overview |
| --------------- | -------- |
[`iree/compiler/embedding_api.h`](https://github.com/openxla/iree/blob/main/compiler/bindings/c/iree/compiler/embedding_api.h) | Top-level IREE compiler embedding API
[`iree/compiler/PluginAPI/` directory](https://github.com/openxla/iree/tree/main/compiler/src/iree/compiler/PluginAPI) | IREE compiler plugin API
[`mlir/include/mlir-c/` directory](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir-c) | MLIR C API headers

### Concepts

The compiler API is centered around running pipelines to translate inputs to
artifacts. These are modeled via _sessions_, _invocations_, _sources_, and
_outputs_.

```mermaid
stateDiagram-v2
  accTitle: IREE compiler session and invocation state diagram
  accDescr {
    Input files are opened (or buffers are wrapped) as sources in a session.
    Sources are parsed into invocations, which run pipelines.
    Output files are written (or buffers are mapped) for compilation artifacts.
  }

  direction LR
  InputFile --> Source1 : open file
  InputBuffer --> Source2 : wrap buffer

  state Session {
    Source1 --> Invocation1
    Source1 --> Invocation2
    Source2 --> Invocation3
    Invocation1 --> Invocation1 : run pipeline
    Invocation2 --> Invocation2 : run pipeline
    Invocation3 --> Invocation3 : run pipeline
  }

  Invocation1 --> OutputFile1   : write file
  Invocation2 --> OutputBuffer1 : map memory
  Invocation3 --> OutputBuffer2 : map memory
```

#### Sessions

A _session_ represents a scope where one or more invocations can be executed.

* Internally, _sessions_ consist of an `MLIRContext` and a private set of
  _options_.
* _Sessions_ may activate available _plugins_ based on their _options_.

#### Invocations

An _invocation_ represents a discrete run of the compiler.

* _Invocations_ run _pipelines_, consisting of _passes_, to translate from
  _sources_ to _outputs_.

#### Sources

A _source_ represents an input program, including operations and data.

* _Sources_ may refer to files or buffers in memory.

#### Outputs

An _output_ represents a compilation artifact.

* _Outputs_ can be standalone files or more advanced streams.

#### Plugins

A _plugin_ extends the compiler with some combination of target backends,
options, passes, or pipelines.

### Quickstart

!!! todo - "Under construction"

    TODO: how to obtain `libIREECompiler.so`, link to a sample/template project

## Runtime API

The IREE runtime is structured as a modular set of library components. Each
component is designed to be linked into applications directly and compiled
with LTO style optimizations.

The low level library components can be used directly or through a higher level
API.

=== "High level API"

    The high level 'runtime' API sits on top of the low level components. It is
    relatively terse but does not expose the full flexibility of the underlying
    systems.

    ```mermaid
    graph TD
      accTitle: IREE runtime high level API diagram
      accDescr {
      The IREE runtime includes 'base', 'HAL', and 'VM' components, each with
      their own types and API methods.
      A high level "runtime API" sits on top of these component APIs.
      Applications can interface indirectly with the IREE runtime via this
      high level runtime API.
      }

      subgraph iree_runtime[IREE Runtime]
        subgraph base
          base_types("Types

          • allocator
          • status
          • etc.")
        end

        subgraph hal[HAL]
          hal_types("Types

          • buffer
          • device
          • etc.")

          hal_drivers("Drivers

          • local-*
          • vulkan
          • etc.")
        end

        subgraph vm[VM]
          vm_types("Types

          • context
          • invocation
          • etc.")
        end

        runtime_api("Runtime API

        • instance
        • session
        • call")

        base_types & hal_types & hal_drivers & vm_types --> runtime_api
      end

      application(Your application)

      runtime_api --> application
    ```

=== "Low level API"

    Each runtime component has its own low level API. The low level APIs are
    typically verbose as they expose the full flexibility of each underlying
    system.

    ```mermaid
    graph TD
      accTitle: IREE runtime low level API diagram
      accDescr {
        The IREE runtime includes 'base', 'HAL', and 'VM' components, each with
        their own types and API methods.
        Applications can interface directly with the IREE runtime via the low
        level component APIs.
      }

      subgraph iree_runtime[IREE Runtime]
        subgraph base
          base_types("Types

          • allocator
          • status
          • etc.")
        end
        subgraph hal[HAL]
          hal_types("Types

          • buffer
          • device
          • etc.")

          hal_drivers("Drivers

          • local-*
          • vulkan
          • etc.")
        end
        subgraph vm[VM]
          vm_types("Types

          • context
          • invocation
          • etc.")
        end
      end

      application(Your application)

      base_types & hal_types & hal_drivers & vm_types --> application
    ```

Runtime API header files are organized by component:

| Component header file | Overview |
| --------------------- | -------- |
[`iree/runtime/api.h`](https://github.com/openxla/iree/blob/main/runtime/src/iree/runtime/api.h) | High level runtime API
[`iree/base/api.h`](https://github.com/openxla/iree/blob/main/runtime/src/iree/base/api.h) | Core API, type definitions, ownership policies, utilities
[`iree/vm/api.h`](https://github.com/openxla/iree/blob/main/runtime/src/iree/vm/api.h) | VM APIs: loading modules, I/O, calling functions
[`iree/hal/api.h`](https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/api.h) | HAL APIs: device management, synchronization, accessing hardware features

### Concepts

By default, IREE uses its own tiny Virtual Machine (VM) at runtime to interpret
program instructions on the host system. VM instructions may also be lowered
further to LLVM IR, C, or other representations for static or resource
constrained deployment.

The VM supports generic operations like loads, stores, arithmetic, function
calls, and control flow. It builds streams of more complex program logic and
dense math into command buffers that are dispatched to hardware backends
through the Hardware Abstraction Layer (HAL) interface.

Most interaction with IREE's C API involves either the VM or the HAL.

<!-- TODO(scotttodd): diagrams -->

<!-- command buffer construction -> dispatch diagram -->
<!-- input buffers -> output buffers diagram -->
<!-- HAL interface diagram -->
<!-- VM module diagram (bytecode, HAL, custom) -->

#### High level - Session

#### High level - Instance

#### High level - Call

#### Low level - VM

* VM _instances_ can serve multiple isolated execution _contexts_
* VM _contexts_ are effectively sandboxes for loading modules and running
  programs
* VM _modules_ provide extra functionality to execution _contexts_, such as
  access to hardware accelerators through the HAL. Compiled user programs are
  also modules.

#### Low level - HAL

* HAL _drivers_ are used to enumerate and create HAL _devices_
* HAL _devices_ interface with hardware, such as by allocating device memory,
  preparing executables, recording and dispatching command buffers, and
  synchronizing with the host
* HAL _buffers_ and _buffer views_ represent storage and shaped/typed views
  into that storage (aka "tensors")

### Quickstart

To use IREE's C API, you will need to build the runtime
[from source](../../building-from-source/getting-started.md). The
[iree-template-cpp](https://github.com/iml130/iree-template-cpp) community
project also shows how to integrate IREE into an external project using
CMake.

The [samples/](https://github.com/openxla/iree/tree/main/samples)
directory demonstrates several ways to use IREE's C API.

!!! todo
    TODO: link to [hello_world.c](https://github.com/benvanik/iree-template-runtime-cmake/blob/main/hello_world.c)

    TODO: link to [hello_world_explained.c](https://github.com/openxla/iree/blob/main/runtime/src/iree/runtime/demo/hello_world_explained.c)

    TODO: link to [simple_embedding.c](https://github.com/openxla/iree/blob/main/samples/simple_embedding/simple_embedding.c)

## Compiler + Runtime = JIT

TODO: `iree-run-mlir`, pjrt plugin, compile "ahead of time" or "just in time"

<!-- ## Advanced usage -->

<!-- TODO(scotttodd): async execution and synchronization -->
<!-- TODO(scotttodd): specialized HAL APIs -->
<!-- TODO(scotttodd): threadpools -->
<!-- TODO(scotttodd): heterogenous execution -->

<!-- ## Troubleshooting -->

<!-- TODO(scotttodd): link to GitHub issues -->
<!-- TODO(scotttodd): common problems? object ownership? loaded modules (HAL)? -->

*[vmfb]: VM FlatBuffer
