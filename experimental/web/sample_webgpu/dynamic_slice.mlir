// This uses push constants.

// Sample usage:
//   iree-compile dynamic_slice_webgpu.mlir \
//     --iree-input-type=mhlo \
//     --iree-hal-target-backends=llvm-cpu \
//     -o ..\iree-tmp\dynamic_slice_llvm_cpu.vmfb
//
//   iree-run-module --module_file=..\iree-tmp\dynamic_slice_llvm_cpu.vmfb \
//     --device=local-task \
//     --entry_function=dynamic_1d_slice \
//     --function_input=4xi32=[1,2,3,4] \
//     --function_input=i32=1
//
//   EXEC @dynamic_1d_slice
//   result[0]: hal.buffer_view
//   2xi32=2 3

// IREE runtime errors:
//   iree_vm_async_invoke_callback_fn_t error:
//   iree/runtime/src/iree/hal/command_buffer_validation.c:80: PERMISSION_DENIED;
//       requested buffer usage is not supported for the buffer on this queue;
//       buffer allows TRANSFER|MAPPING, operation requires TRANSFER (allocator
//       compatibility mismatch); while invoking native function hal.buffer.load;
//       while calling import;
//   [ 1]   native hal.buffer.load:0 -
//   [ 0] bytecode module.dynamic_1d_slice:568 dynamic_slice_webgpu.mlir:19:13
//         at dynamic_slice_webgpu.mlir:18:1

// WebGPU/Dawn errors:
//   [Buffer] usage (BufferUsage::(MapWrite|CopySrc)) doesn't include BufferUsage::CopyDst.
//   - While validating destination [Buffer] usage.
//   - While encoding [CommandEncoder].CopyBufferToBuffer([Buffer], 0, [Buffer], 0, 4).

func.func @dynamic_1d_slice(%input : tensor<4xi32>, %start_indices : tensor<i64>) -> (tensor<2xi32>) {
  %result = "mhlo.dynamic_slice"(%input, %start_indices) {
    slice_sizes = dense<[2]> : tensor<1xi64>
  } : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  return %result : tensor<2xi32>
}
