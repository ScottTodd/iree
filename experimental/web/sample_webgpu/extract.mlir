// This creates no executables (no shaders) and fails when copying buffers.

// iree_vm_async_invoke_callback_fn_t error:
// iree/runtime/src/iree/hal/command_buffer_validation.c:80: PERMISSION_DENIED;
//   requested buffer usage is not supported for the buffer on this queue;
//   buffer allows TRANSFER|MAPPING, operation requires TRANSFER (allocator
//   compatibility mismatch); while invoking native function hal.buffer.load;
//   while calling import;

// 2022-12-12 iree-compile output:
// https://gist.github.com/ScottTodd/7824ba78775b74d92fa2abd3d26eb04b

func.func @extract(%input : tensor<4xi32>, %index : index) -> (i32) {
  %extracted = tensor.extract %input[%index] : tensor<4xi32>
  return %extracted : i32
}
