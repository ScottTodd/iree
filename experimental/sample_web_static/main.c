// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emscripten/emscripten.h>
#include <float.h>
#include <stdio.h>

#include "generated/mnist_bytecode.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode_module.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

typedef struct iree_sample_state_t iree_sample_state_t;
static void iree_sample_state_initialize(iree_sample_state_t* out_state);

// TODO(scotttodd): figure out error handling and state management
//     * out_state and return status would make sense, but emscripten...
iree_sample_state_t* setup_sample();
void cleanup_sample(iree_sample_state_t* state);

int run_sample(iree_sample_state_t* state, float* image_data);

int run_sample_twice(iree_sample_state_t* state, float* image_data) {
  fprintf(stdout, "run first time...\n");
  run_sample(state, image_data);
  fprintf(stdout, "run second time...\n");
  return run_sample(state, image_data);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

extern iree_status_t create_device_with_static_loader(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

typedef struct iree_sample_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_vm_module_t* module;
  iree_runtime_call_t call;
} iree_sample_state_t;

void iree_sample_state_initialize(iree_sample_state_t* out_state) {
  out_state->instance = NULL;
  out_state->device = NULL;
  out_state->module = NULL;
  out_state->session = NULL;
}

iree_status_t create_bytecode_module(iree_vm_module_t** out_module) {
  const struct iree_file_toc_t* module_file_toc = iree_static_mnist_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  return iree_vm_bytecode_module_create(module_data, iree_allocator_null(),
                                        iree_allocator_system(), out_module);
}

iree_sample_state_t* setup_sample() {
  iree_sample_state_t* state;
  state = malloc(sizeof(iree_sample_state_t));
  iree_sample_state_initialize(state);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &state->instance);

  if (iree_status_is_ok(status)) {
    status = create_device_with_static_loader(iree_allocator_system(),
                                              &state->device);
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        state->instance, &session_options, state->device,
        iree_runtime_instance_host_allocator(state->instance), &state->session);
  }

  if (iree_status_is_ok(status)) {
    status = create_bytecode_module(&state->module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(state->session, state->module);
  }

  const char kMainFunctionName[] = "module.predict";
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        state->session, iree_make_cstring_view(kMainFunctionName),
        &state->call);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    free(state);
    return NULL;
  }

  return state;
}

void cleanup_sample(iree_sample_state_t* state) {
  iree_runtime_call_deinitialize(&state->call);

  // Cleanup session and instance.
  iree_hal_device_release(state->device);
  iree_runtime_session_release(state->session);
  iree_runtime_instance_release(state->instance);
  iree_vm_module_release(state->module);

  free(state);
}

int run_sample(iree_sample_state_t* state, float* image_data) {
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_view_t* arg_buffer_view = NULL;
  iree_hal_dim_t buffer_shape[] = {1, 28, 28, 1};
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "iree_hal_buffer_view_allocate_buffer (input)\n");
    // status = iree_hal_buffer_view_allocate_buffer(
    //     iree_hal_device_allocator(state->device), buffer_shape,
    //     IREE_ARRAYSIZE(buffer_shape), IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    //     IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
    //     IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
    //     IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_BUFFER_USAGE_DISPATCH |
    //     IREE_HAL_BUFFER_USAGE_TRANSFER,
    //     iree_make_const_byte_span((void*)image_data, sizeof(float) * 28 *
    //     28), &arg_buffer_view);
    status = iree_hal_buffer_view_wrap_or_clone_heap_buffer(
        iree_hal_device_allocator(state->device), buffer_shape,
        IREE_ARRAYSIZE(buffer_shape), IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
        iree_make_byte_span((void*)image_data, sizeof(float) * 28 * 28),
        iree_allocator_null(), &arg_buffer_view);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "iree_runtime_call_inputs_push_back_buffer_view\n");
    status = iree_runtime_call_inputs_push_back_buffer_view(&state->call,
                                                            arg_buffer_view);
  }
  fprintf(stdout, "iree_hal_buffer_view_release (input)\n");
  iree_hal_buffer_view_release(arg_buffer_view);

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "iree_runtime_call_invoke\n");
    status = iree_runtime_call_invoke(&state->call, /*flags=*/0);
  }

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "iree_runtime_call_outputs_pop_front_buffer_view\n");
    status = iree_runtime_call_outputs_pop_front_buffer_view(&state->call,
                                                             &ret_buffer_view);
  }

  // Read back the results. The output of the mnist model is a 1x10 prediction
  // confidence values for each digit in [0, 9].
  float predictions[10] = {0.0f};
  if (iree_status_is_ok(status)) {
    // fprintf(stdout, "predictions buffer:\n");
    // status = iree_hal_buffer_view_fprint(
    //     stdout, ret_buffer_view,
    //     /*max_element_count=*/4096,
    //     iree_runtime_session_host_allocator(state->session));
    // fprintf(stdout, "\n");

    // TODO(scotttodd): fix "RuntimeError: table index is out of bounds" here
    fprintf(stdout, "iree_hal_buffer_read_data\n");
    status =
        iree_hal_buffer_read_data(iree_hal_buffer_view_buffer(ret_buffer_view),
                                  0, predictions, sizeof(predictions));
  }
  fprintf(stdout, "iree_hal_buffer_view_release (output)\n");
  iree_hal_buffer_view_release(ret_buffer_view);

  fprintf(stdout, "iree_runtime_call_reset (start)\n");
  iree_runtime_call_reset(&state->call);
  fprintf(stdout, "iree_runtime_call_reset (finished)\n");

  if (iree_status_is_ok(status)) {
    // Get the highest index from the output.
    float result_val = FLT_MIN;
    int result_idx = 0;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(predictions); ++i) {
      // fprintf(stdout, "prediction for [%" PRIhsz "]: %f\n", i,
      // predictions[i]);
      if (predictions[i] > result_val) {
        result_val = predictions[i];
        result_idx = i;
      }
    }
    fprintf(stdout, "Detected number: %d\n", result_idx);
    return result_idx;
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return -1;
  }
}
