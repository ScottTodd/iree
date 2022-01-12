// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "generated/simple_mul_bytecode.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode_module.h"

extern iree_status_t create_device_with_static_loader(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

iree_status_t create_bytecode_module(iree_vm_module_t** out_module) {
  const struct iree_file_toc_t* module_file_toc =
      iree_static_simple_mul_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  return iree_vm_bytecode_module_create(module_data, iree_allocator_null(),
                                        iree_allocator_system(), out_module);
}

iree_status_t Run() {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().
  iree_runtime_instance_t* instance = NULL;

  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = create_device_with_static_loader(iree_allocator_system(), &device);
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  iree_vm_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = create_bytecode_module(&module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(session, module);
  }

  const char kMainFunctionName[] = "module.simple_mul";
  iree_runtime_call_t call;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view(kMainFunctionName), &call);
  }

  // Populate initial values for 4 * 2 = 8.
  const int kElementCount = 4;
  iree_hal_dim_t shape[1] = {kElementCount};
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;
  float kFloat4[] = {4.0f, 4.0f, 4.0f, 4.0f};
  float kFloat2[] = {2.0f, 2.0f, 2.0f, 2.0f};

  iree_hal_memory_type_t input_memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_hal_device_allocator(device), shape, IREE_ARRAYSIZE(shape),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        input_memory_type,
        IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
        iree_make_const_byte_span((void*)kFloat4,
                                  sizeof(float) * kElementCount),
        &arg0_buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_hal_device_allocator(device), shape, IREE_ARRAYSIZE(shape),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        input_memory_type,
        IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
        iree_make_const_byte_span((void*)kFloat2,
                                  sizeof(float) * kElementCount),
        &arg1_buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_inputs_push_back_buffer_view(&call, arg0_buffer_view);
  }
  iree_hal_buffer_view_release(arg0_buffer_view);
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_inputs_push_back_buffer_view(&call, arg1_buffer_view);
  }
  iree_hal_buffer_view_release(arg1_buffer_view);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call,
                                                             &ret_buffer_view);
  }

  // Read back the results and ensure we got the right values.
  float results[] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_read_data(iree_hal_buffer_view_buffer(ret_buffer_view),
                                  0, results, sizeof(results));
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
      fprintf(stdout, "result[%" PRIhsz "]: %f\n", i, results[i]);
      if (results[i] != 8.0f) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
        break;
      }
    }
  }

  // Cleanup call and buffers.
  iree_hal_buffer_view_release(ret_buffer_view);
  iree_runtime_call_deinitialize(&call);

  // Cleanup session and instance.
  iree_hal_device_release(device);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  iree_vm_module_release(module);

  return status;
}

#include <math.h>
int int_sqrt(int x) { return sqrt(x); }

// TODO(scotttodd): more interesting demo:
//    - [ ] MNIST or some other program instead of mul()
//    - [ ] a few exported functions to call: init(), predict(), metadata()

int main() {
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }

  fprintf(stdout, "static sample completed successfully\n");
  return 0;
}
