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

By default, the `.vmfb` module files produced by the IREE compiler can be opened
as zip files, allowing inspection of the generated code using other tools.

```console
$ zip -sf simple_abs_cpu.vmfb

  Archive contains:
    module.fb
    abs_dispatch_0_system_elf_x86_64.so
  Total 2 entries (13892 bytes)
```

<!-- TODO(scotttodd): keep the shorter output ^, but use `unzip` to query
                      symbols on the .so
-->
```console
$ unzip -l simple_abs_cpu.vmfb

  Archive:  simple_abs_cpu.vmfb
    Length      Date    Time    Name
  ---------  ---------- -----   ----
      3964  1980-01-01 00:00   module.fb
      9928  1980-01-01 00:00   abs_dispatch_0_system_elf_x86_64.so
  ---------                     -------
      13892                     2 files
```

<!-- TODO(scotttodd): add annotation (insiders only), qualifying "default" with
                      `--iree-vm-emit-polyglot-zip=true`
-->

`iree-dump-module`

7zip (`--iree-vm-emit-polyglot-zip=true`)

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

    ```bash hl_lines="5 6"
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
      module_abs_dispatch_0_system_elf_x86_64.so # (1)!
      simple_abs_cpu.vmfb
    ```

    1.  These are platform-specific files, so this will be
    `*_system_dll_x86_64.dll` on Windows, for example.

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
