# IREE compiler tips and tricks

!!! note ""

    ```
    Planning notes

    Key: [ ] not started; [-] started; [x] finished

    [x] List backend targets
    [x] Compile for each
    [-] Dump sources (content tabs for each target)
    [ ] Inspect .vmfb
    ```

!!! tip "Tip - Options and the Python bindings"

    If you are using the Python bindings, options can be passed via the
    `extra_args=["--flag"]` argument:

    ```python
    compiled_module = iree.compiler.tools.compile_str(
        input_mlir, target_backends=["llvm-cpu"], extra_args=["--mlir-timing"])
    ```

## Target backends

### Choosing a target backend

See which compiler target backends are available:

```console
$ iree-compile --iree-hal-list-target-backends

  Registered target backends:
    llvm-cpu
    metal
    metal-spirv
    vmvx
    vmvx-inline
    vulkan
    vulkan-spirv
```

The [deployment configuration](./deployment-configurations/index.md) guides have
more information about each target backend. Once you have chosen which
backend(s) to use, run the compiler:

!!! info inline end "Multi-targeting"

    Repeat the `--iree-hal-target-backends` flag to build a module containing
    code for multiple targets.

=== "CPU"

    ```console
    $ iree-compile \
      --iree-hal-target-backends=llvm-cpu \
      program.mlir -o program_cpu.vmfb
    ```

=== "GPU - Vulkan"

    ```console
    $ iree-compile \
      --iree-hal-target-backends=vulkan-spirv \
      program.mlir -o program_vulkan.vmfb
    ```

=== "GPU - CUDA"

    ```console
    $ iree-compile \
      --iree-hal-target-backends=cuda \
      program.mlir -o program_cuda.vmfb
    ```

### Setting target backend options

Each target backend can provide it's own set of options. For example, the
`llvm-cpu` backend can generate code for multiple CPU architectures with
different CPU features.

!!! note

    The compiler can usually pick appropriate defaults for the machine where it
    is running, but you may want to compile on one machine and then deploy on
    another device or set of devices.

=== "CPU"

    For the full list of options:

    ```console
    $ iree-compile --help | grep llvmcpu
    ```

=== "GPU - Vulkan"

    For the full list of options:

    ```console
    $ iree-compile --help | grep vulkan
    ```

=== "GPU - CUDA"

    For the full list of options:

    ```console
    $ iree-compile --help | grep cuda
    ```

## Inspecting compiler outputs

??? example "Executable source MLIR"

    ```
    hal.executable public @linear_dispatch_0 {
      hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_nv = []>>}> {
        hal.executable.export public @linear_dispatch_0_generic_128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
        ^bb0(%arg0: !hal.device):
          %x, %y, %z = flow.dispatch.workgroup_count_from_slice
          hal.return %x, %y, %z : index, index, index
        }
        builtin.module {
          func.func @linear_dispatch_0_generic_128_f32() {
            %c0 = arith.constant 0 : index
            %cst = arith.constant 2.000000e+00 : f32
            %cst_0 = arith.constant 3.000000e+00 : f32
            %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
            %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
            %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
            %3 = tensor.empty() : tensor<128xf32>
            %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<128xf32>) outs(%3 : tensor<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %5 = arith.mulf %in, %cst : f32
              %6 = arith.addf %5, %cst_0 : f32
              linalg.yield %6 : f32
            } -> tensor<128xf32>
            flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [128], strides = [1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
            return
          }
        }
      }
    }
    ```

=== "CPU"

    TODO

=== "GPU - Vulkan"

    ```console
    $ iree-compile \
      --iree-hal-target-backends=vulkan-spirv \
      --iree-vulkan-target-triple=pascal-unknown-linux \
      --iree-hal-dump-executable-sources-to=linear_vulkan/ \
      --iree-hal-dump-executable-intermediates-to=linear_vulkan/ \
      --iree-hal-dump-executable-binaries-to=linear_vulkan/ \
      linear.mlir -o linear_vulkan.vmfb
    ```

    Use `spirv-dis` from
    [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) to disassemble
    the binary SPIR-V into assembly language text:

    ```console
    $ spirv-dis \
      linear_vulkan/module_linear_dispatch_0_vulkan_spirv_fb.spv > \
      linear_vulkan/module_linear_dispatch_0_vulkan_spirv_fb.spvasm
    ```

    ??? example "SPIR-V assembly text"

        ```
        ; SPIR-V
        ; Version: 1.0
        ; Generator: Khronos; 22
        ; Bound: 32
        ; Schema: 0
                      OpCapability Shader
                      OpExtension "SPV_KHR_storage_buffer_storage_class"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %linear_dispatch_0_generic_128_f32 "linear_dispatch_0_generic_128_f32" %__builtin_var_WorkgroupId__ %__builtin_var_LocalInvocationId__
                      OpExecutionMode %linear_dispatch_0_generic_128_f32 LocalSize 64 1 1
                      OpName %__builtin_var_LocalInvocationId__ "__builtin_var_LocalInvocationId__"
                      OpName %__builtin_var_WorkgroupId__ "__builtin_var_WorkgroupId__"
                      OpName %__resource_var_0_0_ "__resource_var_0_0_"
                      OpName %__resource_var_0_1_ "__resource_var_0_1_"
                      OpName %linear_dispatch_0_generic_128_f32 "linear_dispatch_0_generic_128_f32"
                      OpDecorate %__builtin_var_LocalInvocationId__ BuiltIn LocalInvocationId
                      OpDecorate %__builtin_var_WorkgroupId__ BuiltIn WorkgroupId
                      OpDecorate %_runtimearr_float ArrayStride 4
                      OpMemberDecorate %_struct_7 0 Offset 0
                      OpDecorate %_struct_7 Block
                      OpDecorate %__resource_var_0_0_ Binding 0
                      OpDecorate %__resource_var_0_0_ DescriptorSet 0
                      OpDecorate %__resource_var_0_1_ Binding 1
                      OpDecorate %__resource_var_0_1_ DescriptorSet 0
              %uint = OpTypeInt 32 0
            %v3uint = OpTypeVector %uint 3
        %_ptr_Input_v3uint = OpTypePointer Input %v3uint
        %__builtin_var_LocalInvocationId__ = OpVariable %_ptr_Input_v3uint Input
        %__builtin_var_WorkgroupId__ = OpVariable %_ptr_Input_v3uint Input
              %float = OpTypeFloat 32
        %_runtimearr_float = OpTypeRuntimeArray %float
          %_struct_7 = OpTypeStruct %_runtimearr_float
        %_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
        %__resource_var_0_0_ = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %__resource_var_0_1_ = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
              %void = OpTypeVoid
                %12 = OpTypeFunction %void
            %uint_64 = OpConstant %uint 64
            %uint_0 = OpConstant %uint 0
            %float_2 = OpConstant %float 2
            %float_3 = OpConstant %float 3
        %_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
        %linear_dispatch_0_generic_128_f32 = OpFunction %void None %12
                %15 = OpLabel
                %20 = OpLoad %v3uint %__builtin_var_WorkgroupId__
                %21 = OpCompositeExtract %uint %20 0
                %22 = OpLoad %v3uint %__builtin_var_LocalInvocationId__
                %23 = OpCompositeExtract %uint %22 0
                %24 = OpIMul %uint %21 %uint_64
                %25 = OpIAdd %uint %23 %24
                %27 = OpAccessChain %_ptr_StorageBuffer_float %__resource_var_0_0_ %uint_0 %25
                %28 = OpLoad %float %27
                %29 = OpFMul %float %28 %float_2
                %30 = OpFAdd %float %29 %float_3
                %31 = OpAccessChain %_ptr_StorageBuffer_float %__resource_var_0_1_ %uint_0 %25
                      OpStore %31 %30
                      OpReturn
                      OpFunctionEnd

        ```

=== "GPU - CUDA"

    TODO
