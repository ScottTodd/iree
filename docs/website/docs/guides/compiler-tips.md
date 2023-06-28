# IREE compiler tips and tricks

!!! note ""

    ```
    Planning notes

    Key: [ ] not started; [-] started; [x] finished

    [ ] MLIR overview?
    [x] Dump sources (content tabs for each target)
    [ ] Inspect .vmfb as zip
    [ ] `--compile-to`
    [ ] dot file export
     ↳ [ ] `--iree-flow-dump-dispatch-graph`
     ↳ [ ] `--view-op-graph`
    [ ] `--iree-flow-trace-dispatch-tensors`

    ```

(Introduce goals and supported workflows here?)

IREE targets a diverse range of hardware platform targets and is built using
modular compiler technologies.......

## Setting compiler options

Tools such as `iree-compile` take options via command-line flags. Pass `--help`
to see the full list:

```console
$ iree-compile --help

OVERVIEW: IREE compilation driver

USAGE: iree-compile [options] <input file or '-' for stdin>

OPTIONS:
  ...
```

!!! tip "Tip - Options and the Python bindings"

    If you are using the Python bindings, options can be passed via the
    `extra_args=["--flag"]` argument:

    ``` python hl_lines="12"
    import iree.compiler as ireec

    input_mlir = """
    func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
      %result = math.absf %input : tensor<f32>
      return %result : tensor<f32>
    }"""

    compiled_module = ireec.tools.compile_str(
        input_mlir,
        target_backends=["llvm-cpu"],
        extra_args=["--mlir-timing"])
    ```

## Inspecting `.vmfb` files

The IREE compiler generates [FlatBuffer](https://flatbuffers.dev/) files using
the `.vmfb` file extension, short for "Virtual Machine FlatBuffer", which can
then be loaded and executed using IREE's runtime. By default, these files can
be opened as zip files:

<!-- TODO(scotttodd): add annotation (insiders only), qualifying "default" with
                      `--iree-vm-emit-polyglot-zip=true`
-->

```console
$ unzip -d simple_abs_cpu ./simple_abs_cpu.vmfb

Archive:  ./simple_abs_cpu.vmfb
  extracting: simple_abs_cpu/module.fb
  extracting: simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so
```

The embedded binary (here an ELF shared object with CPU code) can be parsed by
standard tools:

```console
$ readelf -Ws ./simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so

Symbol table '.dynsym' contains 2 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000001760    17 FUNC    GLOBAL DEFAULT    7 iree_hal_executable_library_query

Symbol table '.symtab' contains 42 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abs_dispatch_0
    2: 0000000000001730    34 FUNC    LOCAL  DEFAULT    7 abs_dispatch_0_generic
    3: 00000000000034c0    80 OBJECT  LOCAL  DEFAULT    8 iree_hal_executable_library_query_v0
    4: 0000000000001780   111 FUNC    LOCAL  DEFAULT    7 iree_h2f_ieee
    5: 00000000000017f0   207 FUNC    LOCAL  DEFAULT    7 iree_f2h_ieee
    ...
```

The `iree-dump-module` tool can also be used to see information about a given
`.vmfb` file:

```console
$ iree-dump-module simple_abs.vmfb

//===--------------------------------------------------------------------------------------------------------------===//
// @module : version 0
//===--------------------------------------------------------------------------------------------------------------===//

Required Types:
  [  0] i32
  [  1] i64
  [  2] !hal.allocator
  [  3] !hal.buffer
  [  4] !hal.buffer_view
  [  5] !hal.command_buffer
  [  6] !hal.descriptor_set_layout
  [  7] !hal.device
  [  8] !hal.executable
  [  9] !hal.fence
  [ 10] !hal.pipeline_layout
  [ 11] !vm.buffer

Module Dependencies:
  hal, version >= 0, required

Imported Functions:
  [  0] hal.ex.shared_device() -> (!vm.ref<?>)
  [  1] hal.allocator.allocate(!vm.ref<?>, i32, i32, i64) -> (!vm.ref<?>)
  [  2] hal.buffer.assert(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>, i64, i32, i32) -> ()
  [  3] hal.buffer_view.create(!vm.ref<?>, i64, i64, i32, i32, tuple<i64>...) -> (!vm.ref<?>)
  [  4] hal.buffer_view.assert(!vm.ref<?>, !vm.ref<?>, i32, i32, tuple<i64>...) -> ()
  [  5] hal.buffer_view.buffer(!vm.ref<?>) -> (!vm.ref<?>)
  [  6] hal.command_buffer.create(!vm.ref<?>, i32, i32, i32) -> (!vm.ref<?>)
  [  7] hal.command_buffer.finalize(!vm.ref<?>) -> ()
  [  8] hal.command_buffer.execution_barrier(!vm.ref<?>, i32, i32, i32) -> ()
  [  9] hal.command_buffer.push_descriptor_set(!vm.ref<?>, !vm.ref<?>, i32, tuple<i32, i32, !vm.ref<?>, i64, i64>...) -> ()
  [ 10] hal.command_buffer.dispatch(!vm.ref<?>, !vm.ref<?>, i32, i32, i32, i32) -> ()
  [ 11] hal.descriptor_set_layout.create(!vm.ref<?>, i32, tuple<i32, i32, i32>...) -> (!vm.ref<?>)
  [ 12] hal.device.allocator(!vm.ref<?>) -> (!vm.ref<?>)
  [ 13] hal.device.query.i64(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>) -> (i32, i64)
  [ 14] hal.device.queue.execute(!vm.ref<?>, i64, !vm.ref<?>, !vm.ref<?>, tuple<!vm.ref<?>>...) -> ()
  [ 15] hal.executable.create(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>, !vm.ref<?>, tuple<!vm.ref<?>>...) -> (!vm.ref<?>)
  [ 16] hal.fence.create(!vm.ref<?>, i32) -> (!vm.ref<?>)
  [ 17] hal.fence.await(i32, tuple<!vm.ref<?>>...) -> (i32)
  [ 18] hal.pipeline_layout.create(!vm.ref<?>, i32, tuple<!vm.ref<?>>...) -> (!vm.ref<?>)

Exported Functions:
  [  0] abs(!vm.ref<?>) -> (!vm.ref<?>)
  [  1] __init() -> ()

//===--------------------------------------------------------------------------------------------------------------===//
// Sections
//===--------------------------------------------------------------------------------------------------------------===//

Module State:
  4 bytes, 2 refs, ~36 bytes total

FlatBuffer: 3964 bytes
  Bytecode: 896 bytes
  .rodata[  0] external     9928 bytes (offset 96 / 60h to 2728h)
  .rodata[  1] embedded       21 bytes `hal.executable.format`
  .rodata[  2] embedded       17 bytes `system-elf-x86_64`
  .rodata[  3] embedded        7 bytes `input 0`
  .rodata[  4] embedded        6 bytes `tensor`

External .rodata: ~9928 bytes

//===--------------------------------------------------------------------------------------------------------------===//
// Bytecode : version 0
//===--------------------------------------------------------------------------------------------------------------===//

  # | Offset   |   Length | Blocks | i32 # | ref # | Requirements | Aliases
----+----------+----------+--------+-------+-------+--------------+-----------------------------------------------------
  0 | 00000000 |      621 |      5 |    20 |     7 |              | abs
  1 | 00000270 |      270 |      4 |     6 |     5 |              | __init

//===--------------------------------------------------------------------------------------------------------------===//
// Debug Information
//===--------------------------------------------------------------------------------------------------------------===//
// NOTE: debug databases are large and should be stripped in deployed artifacts.

Locations: 7
```

`iree-dump-module`

??? info "Info - other output formats"

    The IREE compiler can output multiple formats with the ``--output-format=`
    flag:

    Flag value | Output
    ---------- | ------
    `--output-format=vm-bytecode` (default) | VM Bytecode (`.vmfb`) files
    `--output-format=vm-c` | C source modules

    VM Bytecode files are usable across a range of deployment scenarios, while
    C source modules provide low level connection points for constrained
    environments like bare metal platforms.

## Dumping executable files

The `--iree-hal-dump-executable-*` flags instruct the compiler to save files
related to "executable translation" (code generation for a specific hardware
target) into a directory of your choosing. If you are interested in seeing which
operations in your input program were fused into a compute kernel or what device
code was generated for a given program structure, these flags are a great
starting point.

Flag | Files dumped
---- | ------------
`iree-hal-dump-executable-files-to` | All files (meta-flag)
`iree-hal-dump-executable-sources-to` | Source `.mlir` files prior to HAL compilation
`iree-hal-dump-executable-intermediates-to` | Intermediate files (e.g. `.o` files, `.mlir` stages)
`iree-hal-dump-executable-binaries-to` | Binary files (e.g. `.so`, `.spv`, `.ptx`), as used in the `.vmfb`
`iree-hal-dump-executable-benchmarks-to` | Standalone benchmark files for `iree-benchmark-module`

=== "CPU"

    ```console hl_lines="5 6"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-link-embedded=false \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_cpu.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0.mlir
    module_abs_dispatch_0_system_elf_x86_64_benchmark.mlir
    module_abs_dispatch_0_system_elf_x86_64.codegen.bc
    module_abs_dispatch_0_system_elf_x86_64.linked.bc
    module_abs_dispatch_0_system_elf_x86_64.optimized.bc
    module_abs_dispatch_0_system_elf_x86_64.s
    module_abs_dispatch_0_system_elf_x86_64.so
    simple_abs_cpu.vmfb
    ```

    !!! tip

        The default value of `--iree-llvmcpu-link-embedded=true` generates
        platform-agnostic ELF files. By disabling that flag, the compiler will
        produce `.so` files for Linux, `.dll` files for Windows, etc. While ELF
        files are more portable, inspection of compiled artifacts is easier with
        platform-specific shared object files.

=== "GPU - Vulkan"

    ```console hl_lines="5"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=vulkan-spirv \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_vulkan.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0.mlir
    module_abs_dispatch_0_vulkan_spirv_fb_benchmark.mlir
    module_abs_dispatch_0_vulkan_spirv_fb.mlir
    module_abs_dispatch_0_vulkan_spirv_fb.spv
    simple_abs_vulkan.vmfb
    ```

    !!! tip

        Consider using tools like `spirv-dis` from the
        [SPIR-V Tools project](https://github.com/KhronosGroup/SPIRV-Tools) to
        interact with the `.spv` files.

=== "GPU - CUDA"

    ```console hl_lines="5"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=cuda \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_cuda.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0_cuda_nvptx_fb_benchmark.mlir
    module_abs_dispatch_0_cuda_nvptx_fb.codegen.bc
    module_abs_dispatch_0_cuda_nvptx_fb.linked.bc
    module_abs_dispatch_0_cuda_nvptx_fb.optimized.bc
    module_abs_dispatch_0_cuda_nvptx_fb.ptx
    module_abs_dispatch_0.mlir
    simple_abs_cuda.vmfb
    ```

<!-- TODO(scotttodd): Link to a playground Colab notebook that dumps files -->

## Run phases with `--compile-to`

blah blah blah pipelines stop/resume blah blah
