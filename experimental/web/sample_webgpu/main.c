// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/loop_emscripten.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/buffer.h"
#include "iree/hal/drivers/webgpu/platform/webgpu.h"
#include "iree/hal/drivers/webgpu/webgpu_device.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode/module.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Opaque state for the sample, shared between multiple loaded programs.
typedef struct iree_sample_state_t iree_sample_state_t;

// Initializes the sample and returns its state.
iree_sample_state_t* setup_sample();

// Shuts down the sample and frees its state.
// Requires that all programs first be unloaded with |unload_program|.
void cleanup_sample(iree_sample_state_t* sample_state);

// Opaque state for an individual loaded program.
typedef struct iree_program_state_t iree_program_state_t;

// Loads a program into the sample from the provided data.
// Note: this takes ownership of |vmfb_data|.
iree_program_state_t* load_program(iree_sample_state_t* sample_state,
                                   uint8_t* vmfb_data, size_t length);

// Inspects metadata about a loaded program, printing to stdout.
void inspect_program(iree_program_state_t* program_state);

// Unloads a program and frees its state.
void unload_program(iree_program_state_t* program_state);

// Calls a function synchronously.
//
// Returns a semicolon-delimited list of formatted outputs on success or the
// empty string on failure. Note: This is in need of some real API bindings
// that marshal structured data between C <-> JS.
//
// * |function_name| is the fully qualified function name, like 'module.abs'.
// * |inputs| is a semicolon delimited list of VM scalars and buffers, as
//   described in iree/tooling/vm_util and used in IREE's CLI tools.
//   For example, the CLI `--function_input=f32=1 --function_input=f32=2`
//   should be passed here as `f32=1;f32=2`.
const bool call_function(iree_program_state_t* program_state,
                         const char* function_name, const char* inputs);

//===----------------------------------------------------------------------===//
// Implementation - State and entry points
//===----------------------------------------------------------------------===//

typedef struct iree_sample_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_loop_emscripten_t* loop;
} iree_sample_state_t;

typedef struct iree_program_state_t {
  iree_sample_state_t* sample_state;
  iree_runtime_session_t* session;
  iree_vm_module_t* module;
} iree_program_state_t;

// TODO(scotttodd): explain the flow, state tracking, error handling
//
// call_function
//   parse_inputs_into_call
//   iree_vm_async_invoke
//     invoke_callback
//       process_call_outputs
//         map_call_output[0...n]
//           <queue execute: transfer device -> mappable host>  // move this up
//           wgpuBufferMapAsync
//             buffer_map_async_callback
//               <set event>
//         iree_loop_wait_all
//           map_all_callback
//             <format output JSON and issue callback>

typedef struct iree_call_function_state_t {
  iree_runtime_call_t call;
  iree_loop_emscripten_t* loop;

  // Opaque state used by iree_vm_async_invoke.
  iree_vm_async_invoke_state_t* invoke_state;

  // Timing/statistics metadata.
  iree_time_t invoke_start_time;
  iree_time_t invoke_end_time;
  iree_time_t readback_start_time;
  iree_time_t readback_end_time;

  // Readback state.
  iree_status_t readback_status;  // sticky status for the first async error
  iree_host_size_t outputs_size;
  iree_event_t* readback_events;                         // one per output
  iree_hal_buffer_t** readback_mappable_device_buffers;  // sparse
  iree_hal_buffer_t** readback_mapped_cpu_buffers;       // sparse
} iree_call_function_state_t;

static void iree_call_function_state_destroy(
    iree_call_function_state_t* call_state) {
  fprintf(stderr, "iree_call_function_state_destroy()\n");

  // Readback state.
  for (iree_host_size_t i = 0; i < call_state->outputs_size; ++i) {
    iree_hal_buffer_release(call_state->readback_mapped_cpu_buffers[i]);
  }
  iree_allocator_free(iree_allocator_system(),
                      call_state->readback_mapped_cpu_buffers);
  for (iree_host_size_t i = 0; i < call_state->outputs_size; ++i) {
    iree_hal_buffer_release(call_state->readback_mappable_device_buffers[i]);
  }
  iree_allocator_free(iree_allocator_system(),
                      call_state->readback_mappable_device_buffers);
  for (iree_host_size_t i = 0; i < call_state->outputs_size; ++i) {
    iree_event_deinitialize(&call_state->readback_events[i]);
  }
  iree_allocator_free(iree_allocator_system(), call_state->readback_events);

  iree_status_free(call_state->readback_status);

  // Invoke state.
  iree_allocator_free(iree_allocator_system(), call_state->invoke_state);
  iree_runtime_call_deinitialize(&call_state->call);

  iree_allocator_free(iree_allocator_system(), call_state);
}

extern iree_status_t create_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device);

iree_sample_state_t* setup_sample() {
  iree_sample_state_t* sample_state = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_allocator_system(),
                            sizeof(iree_sample_state_t), (void**)&sample_state);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &sample_state->instance);
  }

  if (iree_status_is_ok(status)) {
    status = create_device(iree_allocator_system(), &sample_state->device);
  }

  if (iree_status_is_ok(status)) {
    status = iree_loop_emscripten_allocate(iree_allocator_system(),
                                           &sample_state->loop);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    cleanup_sample(sample_state);
    return NULL;
  }

  return sample_state;
}

void cleanup_sample(iree_sample_state_t* sample_state) {
  iree_loop_emscripten_free(sample_state->loop);
  iree_hal_device_release(sample_state->device);
  iree_runtime_instance_release(sample_state->instance);
  free(sample_state);
}

iree_program_state_t* load_program(iree_sample_state_t* sample_state,
                                   uint8_t* vmfb_data, size_t length) {
  iree_program_state_t* program_state = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(iree_program_state_t),
                                               (void**)&program_state);
  program_state->sample_state = sample_state;

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        sample_state->instance, &session_options, sample_state->device,
        iree_runtime_instance_host_allocator(sample_state->instance),
        &program_state->session);
  }

  if (iree_status_is_ok(status)) {
    // Take ownership of the FlatBuffer data so JavaScript doesn't need to
    // explicitly call `Module._free()`.
    status = iree_vm_bytecode_module_create(
        iree_runtime_instance_vm_instance(sample_state->instance),
        iree_make_const_byte_span(vmfb_data, length),
        /*flatbuffer_allocator=*/iree_allocator_system(),
        iree_allocator_system(), &program_state->module);
  } else {
    // Must clean up the FlatBuffer data directly.
    iree_allocator_free(iree_allocator_system(), (void*)vmfb_data);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(program_state->session,
                                                program_state->module);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    unload_program(program_state);
    return NULL;
  }

  return program_state;
}

void inspect_program(iree_program_state_t* program_state) {
  fprintf(stdout, "=== program properties ===\n");

  iree_vm_module_t* module = program_state->module;
  iree_string_view_t module_name = iree_vm_module_name(module);
  fprintf(stdout, "  module name: '%.*s'\n", (int)module_name.size,
          module_name.data);

  iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(module);
  fprintf(stdout, "  module signature:\n");
  fprintf(stdout, "    %" PRIhsz " imported functions\n",
          module_signature.import_function_count);
  fprintf(stdout, "    %" PRIhsz " exported functions\n",
          module_signature.export_function_count);
  fprintf(stdout, "    %" PRIhsz " internal functions\n",
          module_signature.internal_function_count);

  fprintf(stdout, "  exported functions:\n");
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    iree_vm_function_t function;
    iree_status_t status = iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      continue;
    }

    iree_string_view_t function_name = iree_vm_function_name(&function);
    iree_vm_function_signature_t function_signature =
        iree_vm_function_signature(&function);
    iree_string_view_t calling_convention =
        function_signature.calling_convention;
    fprintf(stdout, "    function name: '%.*s', calling convention: %.*s'\n",
            (int)function_name.size, function_name.data,
            (int)calling_convention.size, calling_convention.data);
  }
}

void unload_program(iree_program_state_t* program_state) {
  iree_vm_module_release(program_state->module);
  iree_runtime_session_release(program_state->session);
  free(program_state);
}

//===----------------------------------------------------------------------===//
// Input parsing
//===----------------------------------------------------------------------===//

static iree_status_t parse_input_into_call(
    iree_runtime_call_t* call, iree_hal_allocator_t* device_allocator,
    iree_string_view_t input) {
  bool has_equal =
      iree_string_view_find_char(input, '=', 0) != IREE_STRING_VIEW_NPOS;
  bool has_x =
      iree_string_view_find_char(input, 'x', 0) != IREE_STRING_VIEW_NPOS;
  if (has_equal || has_x) {
    // Buffer view (either just a shape or a shape=value) or buffer.
    bool is_storage_reference =
        iree_string_view_consume_prefix(&input, iree_make_cstring_view("&"));
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_parse(input, device_allocator, &buffer_view),
        "parsing value '%.*s'", (int)input.size, input.data);
    if (is_storage_reference) {
      // Storage buffer reference; just take the storage for the buffer view -
      // it'll still have whatever contents were specified (or 0) but we'll
      // discard the metadata.
      iree_vm_ref_t buffer_ref =
          iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
      iree_hal_buffer_view_release(buffer_view);
      return iree_vm_list_push_ref_move(call->inputs, &buffer_ref);
    } else {
      iree_vm_ref_t buffer_view_ref =
          iree_hal_buffer_view_move_ref(buffer_view);
      return iree_vm_list_push_ref_move(call->inputs, &buffer_view_ref);
    }
  } else {
    // Scalar.
    bool has_dot =
        iree_string_view_find_char(input, '.', 0) != IREE_STRING_VIEW_NPOS;
    iree_vm_value_t val;
    if (has_dot) {
      // Float.
      val = iree_vm_value_make_f32(0.0f);
      if (!iree_string_view_atof(input, &val.f32)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parsing value '%.*s' as f32", (int)input.size,
                                input.data);
      }
    } else {
      // Integer.
      val = iree_vm_value_make_i32(0);
      if (!iree_string_view_atoi_int32(input, &val.i32)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parsing value '%.*s' as i32", (int)input.size,
                                input.data);
      }
    }
    return iree_vm_list_push_value(call->inputs, &val);
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Unhandled function input (unreachable?)");
}

static iree_status_t parse_inputs_into_call(
    iree_runtime_call_t* call, iree_hal_allocator_t* device_allocator,
    iree_string_view_t inputs) {
  if (inputs.size == 0) return iree_ok_status();

  // Inputs are provided in a semicolon-delimited list.
  // Split inputs from the list until no semicolons are left.
  iree_string_view_t remaining_inputs = inputs;
  intptr_t split_index = 0;
  do {
    iree_string_view_t next_input;
    split_index = iree_string_view_split(remaining_inputs, ';', &next_input,
                                         &remaining_inputs);
    IREE_RETURN_IF_ERROR(
        parse_input_into_call(call, device_allocator, next_input));
  } while (split_index != -1);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Output readback and formatting
//===----------------------------------------------------------------------===//

typedef struct iree_buffer_map_userdata_v0_t {
  iree_hal_buffer_view_t* source_buffer_view;
  iree_hal_buffer_t* readback_buffer;
} iree_buffer_map_userdata_v0_t;

typedef struct iree_buffer_map_userdata_t {
  iree_call_function_state_t* call_state;
  iree_host_size_t buffer_index;
} iree_buffer_map_userdata_t;

static void iree_webgpu_mapped_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  WGPUBuffer buffer_handle = (WGPUBuffer)user_data;
  wgpuBufferUnmap(buffer_handle);
}

// TODO(scotttodd): move async mapping into webgpu/buffer.h/.c?
static void buffer_map_async_callback_v0(WGPUBufferMapAsyncStatus map_status,
                                         void* userdata_ptr) {
  iree_buffer_map_userdata_v0_t* userdata =
      (iree_buffer_map_userdata_v0_t*)userdata_ptr;
  switch (map_status) {
    case WGPUBufferMapAsyncStatus_Success:
      break;
    case WGPUBufferMapAsyncStatus_Error:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: Error\n");
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: DeviceLost\n");
      break;
    case WGPUBufferMapAsyncStatus_Unknown:
    default:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: Unknown\n");
      break;
  }

  if (map_status != WGPUBufferMapAsyncStatus_Success) {
    iree_hal_buffer_view_release(userdata->source_buffer_view);
    iree_hal_buffer_release(userdata->readback_buffer);
    iree_allocator_free(iree_allocator_system(), userdata);
    return;
  }

  iree_status_t status = iree_ok_status();

  // TODO(scotttodd): bubble result(s) up to the caller (async + callback API)

  iree_device_size_t data_offset = iree_hal_buffer_byte_offset(
      iree_hal_buffer_view_buffer(userdata->source_buffer_view));
  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(userdata->source_buffer_view);
  WGPUBuffer buffer_handle =
      iree_hal_webgpu_buffer_handle(userdata->readback_buffer);

  // For this sample we want to print arbitrary buffers, which is easiest
  // using the |iree_hal_buffer_view_format| function. Internally, that
  // function requires synchronous buffer mapping, so we'll first wrap the
  // already (async) mapped GPU memory into a heap buffer. In a less general
  // application (or one not requiring pretty logging like this), we could
  // skip a few buffer copies and other data transformations here.

  const void* data_ptr =
      wgpuBufferGetConstMappedRange(buffer_handle, data_offset, data_length);

  iree_hal_buffer_t* heap_buffer = NULL;
  if (iree_status_is_ok(status)) {
    // The buffer we get from WebGPU may not be aligned to 64.
    iree_hal_memory_access_t memory_access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_UNALIGNED;
    status = iree_hal_heap_buffer_wrap(
        userdata->readback_buffer->device_allocator,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL, memory_access,
        IREE_HAL_BUFFER_USAGE_MAPPING, data_length,
        iree_make_byte_span((void*)data_ptr, data_length),
        (iree_hal_buffer_release_callback_t){
            .fn = iree_webgpu_mapped_buffer_release,
            .user_data = buffer_handle,
        },
        &heap_buffer);
  }

  // Copy the original buffer_view, backed by the mapped heap buffer instead.
  iree_hal_buffer_view_t* heap_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_create_like(
        heap_buffer, userdata->source_buffer_view, iree_allocator_system(),
        &heap_buffer_view);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Call output:\n");
    status = iree_hal_buffer_view_fprint(stdout, heap_buffer_view,
                                         /*max_element_count=*/4096,
                                         iree_allocator_system());
    fprintf(stdout, "\n");
  }
  iree_hal_buffer_view_release(heap_buffer_view);
  iree_hal_buffer_release(heap_buffer);

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "buffer_map_async_callback_v0 error:\n");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  iree_hal_buffer_view_release(userdata->source_buffer_view);
  iree_hal_buffer_release(userdata->readback_buffer);
  iree_allocator_free(iree_allocator_system(), userdata);
}

static void buffer_map_async_callback(WGPUBufferMapAsyncStatus map_status,
                                      void* userdata_ptr) {
  iree_buffer_map_userdata_t* userdata =
      (iree_buffer_map_userdata_t*)userdata_ptr;
  switch (map_status) {
    case WGPUBufferMapAsyncStatus_Success:
      break;
    case WGPUBufferMapAsyncStatus_Error:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: Error\n");
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: DeviceLost\n");
      break;
    case WGPUBufferMapAsyncStatus_Unknown:
    default:
      fprintf(stderr, "  buffer_map_async_callback_v0 status: Unknown\n");
      break;
  }

  if (map_status != WGPUBufferMapAsyncStatus_Success) {
    // TODO(scotttodd): userdata->call_state->readback_status if unset
    // iree_hal_buffer_view_release(userdata->source_buffer_view);
    // iree_hal_buffer_release(userdata->readback_buffer);
    iree_allocator_free(iree_allocator_system(), userdata);
    return;
  }

  fprintf(stderr, "buffer_map_async_callback[%d] success\n",
          (int)userdata->buffer_index);

  iree_event_set(
      &userdata->call_state->readback_events[userdata->buffer_index]);

  // iree_status_t status = iree_ok_status();

  // // TODO(scotttodd): bubble result(s) up to the caller (async + callback
  // API)

  // iree_device_size_t data_offset = iree_hal_buffer_byte_offset(
  //     iree_hal_buffer_view_buffer(userdata->source_buffer_view));
  // iree_device_size_t data_length =
  //     iree_hal_buffer_view_byte_length(userdata->source_buffer_view);
  // WGPUBuffer buffer_handle =
  //     iree_hal_webgpu_buffer_handle(userdata->readback_buffer);

  // // For this sample we want to print arbitrary buffers, which is easiest
  // // using the |iree_hal_buffer_view_format| function. Internally, that
  // // function requires synchronous buffer mapping, so we'll first wrap the
  // // already (async) mapped GPU memory into a heap buffer. In a less general
  // // application (or one not requiring pretty logging like this), we could
  // // skip a few buffer copies and other data transformations here.

  // const void* data_ptr =
  //     wgpuBufferGetConstMappedRange(buffer_handle, data_offset, data_length);

  // iree_hal_buffer_t* heap_buffer = NULL;
  // if (iree_status_is_ok(status)) {
  //   // The buffer we get from WebGPU may not be aligned to 64.
  //   iree_hal_memory_access_t memory_access =
  //       IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_UNALIGNED;
  //   status = iree_hal_heap_buffer_wrap(
  //       userdata->readback_buffer->device_allocator,
  //       IREE_HAL_MEMORY_TYPE_HOST_LOCAL, memory_access,
  //       IREE_HAL_BUFFER_USAGE_MAPPING, data_length,
  //       iree_make_byte_span((void*)data_ptr, data_length),
  //       (iree_hal_buffer_release_callback_t){
  //           .fn = iree_webgpu_mapped_buffer_release,
  //           .user_data = buffer_handle,
  //       },
  //       &heap_buffer);
  // }

  // // Copy the original buffer_view, backed by the mapped heap buffer instead.
  // iree_hal_buffer_view_t* heap_buffer_view = NULL;
  // if (iree_status_is_ok(status)) {
  //   status = iree_hal_buffer_view_create_like(
  //       heap_buffer, userdata->source_buffer_view, iree_allocator_system(),
  //       &heap_buffer_view);
  // }

  // if (iree_status_is_ok(status)) {
  //   fprintf(stdout, "Call output:\n");
  //   status = iree_hal_buffer_view_fprint(stdout, heap_buffer_view,
  //                                        /*max_element_count=*/4096,
  //                                        iree_allocator_system());
  //   fprintf(stdout, "\n");
  // }
  // iree_hal_buffer_view_release(heap_buffer_view);
  // iree_hal_buffer_release(heap_buffer);

  // if (!iree_status_is_ok(status)) {
  //   fprintf(stderr, "buffer_map_async_callback_v0 error:\n");
  //   iree_status_fprint(stderr, status);
  //   iree_status_free(status);
  // }

  // iree_hal_buffer_view_release(userdata->source_buffer_view);
  // iree_hal_buffer_release(userdata->readback_buffer);
  iree_allocator_free(iree_allocator_system(), userdata);
}

static iree_status_t print_buffer_view(iree_hal_device_t* device,
                                       iree_hal_buffer_view_t* buffer_view) {
  fprintf(stderr, "print_buffer_view\n");
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_device_size_t data_offset = iree_hal_buffer_byte_offset(buffer);
  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // ----------------------------------------------
  // Allocate mappable host memory.
  // Note: iree_hal_webgpu_simple_allocator_allocate_buffer only supports
  // CopySrc today, so we'll create the buffer directly with
  // wgpuDeviceCreateBuffer and then wrap it using iree_hal_webgpu_buffer_wrap.
  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = "IREE_readback",
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size = data_length,
      .mappedAtCreation = false,
  };
  WGPUBuffer readback_buffer_handle = NULL;
  if (iree_status_is_ok(status)) {
    // Note: wgpuBufferDestroy is called after iree_hal_webgpu_buffer_wrap ->
    //       iree_hal_buffer_release -> iree_hal_webgpu_buffer_destroy
    readback_buffer_handle = wgpuDeviceCreateBuffer(
        iree_hal_webgpu_device_handle(device), &descriptor);
    if (!readback_buffer_handle) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "unable to allocate buffer of size %" PRIdsz,
                                data_length);
    }
  }
  iree_device_size_t target_offset = 0;
  const iree_hal_buffer_params_t target_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_hal_buffer_t* readback_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_buffer_wrap(
        device, iree_hal_device_allocator(device), target_params.type,
        target_params.access, target_params.usage, data_length,
        /*byte_offset=*/0,
        /*byte_length=*/data_length, readback_buffer_handle,
        iree_allocator_system(), &readback_buffer);
  }
  // ----------------------------------------------

  // ----------------------------------------------
  // Transfer from device memory to mappable host memory.
  const iree_hal_transfer_command_t transfer_command = {
      .type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY,
      .copy =
          {
              .source_buffer = buffer,
              .source_offset = data_offset,
              .target_buffer = readback_buffer,
              .target_offset = target_offset,
              .length = data_length,
          },
  };
  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_transfer_command_buffer(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, /*transfer_count=*/1, &transfer_command,
        &command_buffer);
  }
  iree_hal_semaphore_t* fence_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, 0ull, &fence_semaphore);
  }
  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &fence_semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, 1, &command_buffer);
  }
  fprintf(stderr, "print_buffer_view after queue_execute, calling wait\n");
  // TODO(scotttodd): Make this async - pass a wait source to iree_loop_wait_one
  // TODO(scotttodd): Ask Ben how to interop between wait sources and HAL
  //                  semaphores. nop_semaphore.c -> promise_semaphore.c?
  //                  Is a semaphore wait even needed? buffer map async might be
  //                  waiting
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(fence_semaphore, signal_value,
                                     iree_infinite_timeout());
  }
  fprintf(stderr, "print_buffer_view after wait\n");
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(fence_semaphore);
  // ----------------------------------------------

  iree_buffer_map_userdata_v0_t* userdata = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(iree_buffer_map_userdata_v0_t),
                                   (void**)&userdata);
    iree_hal_buffer_view_retain(buffer_view);  // Released in the callback.
    userdata->source_buffer_view = buffer_view;
    userdata->readback_buffer = readback_buffer;
  }

  if (iree_status_is_ok(status)) {
    wgpuBufferMapAsync(readback_buffer_handle, WGPUMapMode_Read, /*offset=*/0,
                       /*size=*/data_length, buffer_map_async_callback_v0,
                       /*userdata=*/userdata);
  }

  return status;
}

static iree_status_t print_outputs_from_call(
    iree_runtime_call_t* call, iree_string_builder_t* outputs_builder) {
  iree_vm_list_t* variants_list = iree_runtime_call_outputs(call);
  fprintf(stderr, "print_outputs_from_call, outputs, outputs size: %d\n",
          (int)iree_vm_list_size(variants_list));
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variants_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(variants_list, i, &variant),
        "variant %" PRIhsz " not present", i);

    if (iree_vm_variant_is_value(variant)) {
      switch (iree_vm_type_def_as_value(variant.type)) {
        case IREE_VM_VALUE_TYPE_I8: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i8=%" PRIi8, variant.i8));
          break;
        }
        case IREE_VM_VALUE_TYPE_I16: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i16=%" PRIi16, variant.i16));
          break;
        }
        case IREE_VM_VALUE_TYPE_I32: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i32=%" PRIi32, variant.i32));
          break;
        }
        case IREE_VM_VALUE_TYPE_I64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i64=%" PRIi64, variant.i64));
          break;
        }
        case IREE_VM_VALUE_TYPE_F32: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "f32=%f", variant.f32));
          break;
        }
        case IREE_VM_VALUE_TYPE_F64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "f64=%lf", variant.f64));
          break;
        }
        default: {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(outputs_builder, "?"));
          break;
        }
      }
    } else if (iree_vm_variant_is_ref(variant)) {
      if (iree_hal_buffer_view_isa(variant.ref)) {
        iree_hal_buffer_view_t* buffer_view =
            iree_hal_buffer_view_deref(variant.ref);
        // TODO(scotttodd): join async outputs together and return to caller
        iree_hal_device_t* device = iree_runtime_session_device(call->session);
        IREE_RETURN_IF_ERROR(print_buffer_view(device, buffer_view));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            outputs_builder, "(no printer)"));
      }
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(outputs_builder, "(null)"));
    }

    if (i < iree_vm_list_size(variants_list) - 1) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(outputs_builder, ";"));
    }
  }

  iree_vm_list_resize(variants_list, 0);

  return iree_ok_status();
}

static iree_status_t map_all_callback(void* user_data, iree_loop_t loop,
                                      iree_status_t status) {
  fprintf(stderr, "iree_loop_wait_all callback\n");

  iree_call_function_state_t* call_state =
      (iree_call_function_state_t*)user_data;

  call_state->readback_end_time = iree_time_now();

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "iree_vm_async_invoke_callback_fn_t error:\n");
    iree_status_fprint(stderr, status);
    // Note: loop_emscripten.js must free 'status'!
  }

  iree_call_function_state_destroy(call_state);
  return status;
}

static iree_status_t allocate_mappable_device_buffer(
    iree_hal_device_t* device, iree_hal_buffer_view_t* buffer_view,
    iree_hal_buffer_t** out_buffer) {
  fprintf(stderr, "    allocate_mappable_device_buffer\n");
  *out_buffer = NULL;

  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // Note: iree_hal_webgpu_simple_allocator_allocate_buffer only supports
  // CopySrc today, so we'll create the buffer directly with
  // wgpuDeviceCreateBuffer and then wrap it using iree_hal_webgpu_buffer_wrap.
  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = "IREE_readback",
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size = data_length,
      .mappedAtCreation = false,
  };
  WGPUBuffer readback_buffer_handle = NULL;
  // Note: wgpuBufferDestroy is called after iree_hal_webgpu_buffer_wrap ->
  //       iree_hal_buffer_release -> iree_hal_webgpu_buffer_destroy
  readback_buffer_handle = wgpuDeviceCreateBuffer(
      iree_hal_webgpu_device_handle(device), &descriptor);
  if (!readback_buffer_handle) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "unable to allocate buffer of size %" PRIdsz,
                            data_length);
  }
  const iree_hal_buffer_params_t target_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  return iree_hal_webgpu_buffer_wrap(
      device, iree_hal_device_allocator(device), target_params.type,
      target_params.access, target_params.usage, data_length,
      /*byte_offset=*/0,
      /*byte_length=*/data_length, readback_buffer_handle,
      iree_allocator_system(), out_buffer);
}

// Processes outputs from a completed function invocation.
// Some output data types may require asynchronous mapping (readback).
static iree_status_t process_call_outputs(
    iree_call_function_state_t* call_state) {
  call_state->readback_start_time = iree_time_now();

  iree_vm_list_t* outputs_list = iree_runtime_call_outputs(&call_state->call);
  iree_host_size_t outputs_size = iree_vm_list_size(outputs_list);
  fprintf(stderr, "process_call_outputs, outputs size: %d\n",
          (int)outputs_size);
  iree_hal_device_t* device =
      iree_runtime_session_device(call_state->call.session);

  // See loop_test.h WaitOneBlocking / WaitAllBlocking
  // list of `iree_event_t`s that can be set later
  //   start set if no readback required
  // list of `iree_wait_source`s (one per event)
  // iree_loop_wait_all(wait_sources, format_outputs_callback)
  // for each ref / buffer_view
  //   readback(event_to_set_on_completion)
  //     use loop as needed

  // Allocate lists. Note: empty object contents, may be sparse.
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(iree_event_t) * outputs_size,
      (void**)&call_state->readback_events));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(iree_hal_buffer_t*) * outputs_size,
      (void**)&call_state->readback_mappable_device_buffers));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(iree_hal_buffer_t*) * outputs_size,
      (void**)&call_state->readback_mapped_cpu_buffers));
  // Note: setting the size after mallocs so the destroy() function doesn't try
  // to release from uninitialized memory if allocation failed.
  call_state->outputs_size = outputs_size;

  iree_wait_source_t* wait_sources = (iree_wait_source_t*)iree_alloca(
      sizeof(iree_wait_source_t) * outputs_size);

  // Loop through the outputs once to find buffers that need readback.
  iree_host_size_t buffer_count = 0;
  for (iree_host_size_t i = 0; i < outputs_size; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(outputs_list, i, &variant),
        "variant %" PRIhsz " not present", i);

    if (iree_vm_variant_is_ref(variant)) {
      fprintf(stderr, "  [%" PRIhsz "]: ref\n", i);
      if (!iree_hal_buffer_view_isa(variant.ref)) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "only buffer_view variants are supported");
      }

      // Output is a buffer_view ref, add to readback batch (async).
      iree_event_initialize(false, &call_state->readback_events[i]);
      buffer_count++;
      iree_hal_buffer_view_t* buffer_view =
          iree_hal_buffer_view_deref(variant.ref);
      IREE_RETURN_IF_ERROR(allocate_mappable_device_buffer(
          device, buffer_view,
          &call_state->readback_mappable_device_buffers[i]));
    } else {
      // Not a buffer_view ref, data is available immediately - start signaled.
      fprintf(stderr, "  [%" PRIhsz "]: other\n", i);
      iree_event_initialize(true, &call_state->readback_events[i]);
    }
    wait_sources[i] = iree_event_await(&call_state->readback_events[i]);
  }

  fprintf(stderr, "  buffer_count: %d\n", (int)buffer_count);

  // Loop through the outputs again to build a batched transfer command buffer.
  iree_hal_transfer_command_t* transfer_commands =
      (iree_hal_transfer_command_t*)iree_alloca(
          sizeof(iree_hal_transfer_command_t) * buffer_count);
  for (iree_host_size_t i = 0, buffer_index = 0; i < outputs_size; ++i) {
    // TODO(scotttodd): Track buffers some other way... lots of duplicate code
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(outputs_list, i, &variant),
        "variant %" PRIhsz " not present", i);
    if (!iree_vm_variant_is_ref(variant)) continue;
    if (!iree_hal_buffer_view_isa(variant.ref)) continue;

    iree_hal_buffer_view_t* buffer_view =
        iree_hal_buffer_view_deref(variant.ref);

    iree_hal_buffer_t* source_buffer = iree_hal_buffer_view_buffer(buffer_view);
    iree_device_size_t data_offset = iree_hal_buffer_byte_offset(source_buffer);
    iree_hal_buffer_t* readback_buffer =
        call_state->readback_mappable_device_buffers[i];
    iree_device_size_t target_offset = 0;
    iree_device_size_t data_length =
        iree_hal_buffer_view_byte_length(buffer_view);
    transfer_commands[buffer_index] = (iree_hal_transfer_command_t){
        .type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY,
        .copy =
            {
                .source_buffer = source_buffer,
                .source_offset = data_offset,
                .target_buffer = readback_buffer,
                .target_offset = target_offset,
                .length = data_length,
            },
    };
    buffer_index++;
  }

  // Construct and issue the transfer command buffer, then wait on it.
  iree_status_t status = iree_ok_status();
  iree_hal_command_buffer_t* transfer_command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_transfer_command_buffer(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, buffer_count, transfer_commands,
        &transfer_command_buffer);
  }
  iree_hal_semaphore_t* signal_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, 0ull, &signal_semaphore);
  }
  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &signal_semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, 1, &transfer_command_buffer);
  }
  fprintf(stderr, "  process_call_outputs after queue_execute, calling wait\n");
  // TODO(scotttodd): Make this async - pass a wait source to iree_loop_wait_one
  //     1. create iree_hal_fence_t, iree_hal_fence_insert(fance, semaphore)
  //     2. iree_hal_fence_await -> iree_wait_source_t
  //     3. iree_loop_wait_one(loop, wait_source, ...)
  //   (requires moving off of nop_semaphore and wait source import)
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                     iree_infinite_timeout());
  }
  fprintf(stderr, "  process_call_outputs after wait\n");
  iree_hal_command_buffer_release(transfer_command_buffer);
  iree_hal_semaphore_release(signal_semaphore);

  // wgpuBufferMapAsync the mappable device buffers
  for (iree_host_size_t i = 0; i < outputs_size; ++i) {
    if (!iree_status_is_ok(status)) break;
    if (!call_state->readback_mappable_device_buffers[i]) continue;

    fprintf(stderr, "  [%d], wgpuBufferMapAsync\n", (int)i);
    iree_buffer_map_userdata_t* map_userdata = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), sizeof(iree_buffer_map_userdata_t),
        (void**)&map_userdata));

    // TODO(scotttodd): Track buffers some other way... lots of duplicate code
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(outputs_list, i, &variant),
        "variant %" PRIhsz " not present", i);
    iree_hal_buffer_view_t* buffer_view =
        iree_hal_buffer_view_deref(variant.ref);
    iree_hal_buffer_view_retain(buffer_view);  // Released in the callback.

    map_userdata->call_state = call_state;
    map_userdata->buffer_index = i;
    iree_device_size_t data_length =
        iree_hal_buffer_view_byte_length(buffer_view);

    WGPUBuffer device_buffer = iree_hal_webgpu_buffer_handle(
        call_state->readback_mappable_device_buffers[i]);
    wgpuBufferMapAsync(device_buffer, WGPUMapMode_Read,
                       /*offset=*/0,
                       /*size=*/data_length, buffer_map_async_callback,
                       /*userdata=*/map_userdata);
  }

  // iree_buffer_map_userdata_v0_t* userdata = NULL;
  // if (iree_status_is_ok(status)) {
  //   status = iree_allocator_malloc(iree_allocator_system(),
  //                                  sizeof(iree_buffer_map_userdata_v0_t),
  //                                  (void**)&userdata);
  //   iree_hal_buffer_view_retain(buffer_view);  // Released in the callback.
  //   userdata->source_buffer_view = buffer_view;
  //   userdata->readback_buffer = readback_buffer;
  // }

  // if (iree_status_is_ok(status)) {
  //   wgpuBufferMapAsync(readback_buffer_handle, WGPUMapMode_Read,
  //   /*offset=*/0,
  //                      /*size=*/data_length, buffer_map_async_callback_v0,
  //                      /*userdata=*/userdata);
  // }

  // TODO(scotttodd): one transfer command buffer / queue execute for all
  //                  buffers, then separately issue wgpuBufferMapAsync calls

  // WORKING HERE
  //
  // state:
  //   events[]
  //   readback_mappable_device_buffers[]
  //   readback_mapped_cpu_buffers[]
  //   status
  //
  // for each buffer output
  //   wgpuDeviceCreateBuffer -> WGPUBuffer
  //   iree_hal_webgpu_buffer_wrap -> readback_mappable_device_buffers[i]
  //   iree_hal_create_transfer_command_buffer
  // wait for transfer
  // for each buffer output
  //   wgpuBufferMapAsync
  //     userdata: iree_call_function_state_t + index
  //   callback
  //     failure -> sticky failure on iree_call_function_state_t?
  //       if existing status is not ok, ignore
  //       if existing status _is_ ok, transfer over
  //     wgpuBufferGetConstMappedRange
  //     iree_hal_heap_buffer_wrap -> readback_mapped_cpu_buffers[i]
  //     signal event [i]
  //
  // WORKING HERE

  IREE_RETURN_IF_ERROR(iree_loop_wait_all(
      iree_loop_emscripten(call_state->loop), outputs_size, wait_sources,
      iree_make_timeout_ms(1000), map_all_callback,
      /*user_data=*/call_state));

  // fprintf(stderr, "  [hack] setting readback_events[0]\n");
  // iree_event_set(&call_state->readback_events[0]);  // DO NOT SUBMIT

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function calling / invocations
//===----------------------------------------------------------------------===//

// Handles the completion callback from `iree_vm_async_invoke()`.
iree_status_t invoke_callback(void* user_data, iree_loop_t loop,
                              iree_status_t status, iree_vm_list_t* outputs) {
  fprintf(stderr, "iree_vm_async_invoke_callback_fn_t\n");
  iree_call_function_state_t* call_state =
      (iree_call_function_state_t*)user_data;

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "iree_vm_async_invoke_callback_fn_t error:\n");
    iree_status_fprint(stderr, status);
    iree_call_function_state_destroy(call_state);
    return status;  // Note: loop_emscripten.js must free this!
  }

  //
  call_state->invoke_end_time = iree_time_now();

  // ----------------------- remove after debugging
  iree_time_t elapsed_time =
      call_state->invoke_end_time - call_state->invoke_start_time;
  iree_time_t elapsed_time_millis = elapsed_time / 1000000;
  fprintf(stderr, "  elapsed time: %dms\n", (int)elapsed_time_millis);
  // TODO(scotttodd): return this to JS
  // ----------------------- remove after debugging

  status = process_call_outputs(call_state);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "process_call_outputs error:\n");
    iree_status_fprint(stderr, status);
    iree_call_function_state_destroy(call_state);
    // Note: loop_emscripten.js must free 'status'!
  } else {
    // Do _not_ deinitialize/free call_state, async callbacks need it.
  }

  return status;
}

// TODO(scotttodd): return a Promise that resolves
//   ... with the output string?
//   ... with nothing (then query for output data?)
// create wait handle, return the Promise
//   set the wait primitive to resolve? can't pass a value via iree_event_set?
// receive a function to call when complete? Promise API would be cleaner
// internal implementation so could just call a function with "" or real data
// or pass sucess/failure functions to call

const bool call_function(iree_program_state_t* program_state,
                         const char* function_name, const char* inputs) {
  iree_status_t status = iree_ok_status();

  iree_call_function_state_t* call_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(iree_call_function_state_t),
                                   (void**)&call_state);
  }
  call_state->loop = program_state->sample_state->loop;

  // Fully qualify the function name. This sample only supports loading one
  // module (i.e. 'program') per session, so we can do this.
  iree_string_builder_t name_builder;
  iree_string_builder_initialize(iree_allocator_system(), &name_builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t module_name = iree_vm_module_name(program_state->module);
    status = iree_string_builder_append_format(&name_builder, "%.*s.%s",
                                               (int)module_name.size,
                                               module_name.data, function_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        program_state->session, iree_string_builder_view(&name_builder),
        &call_state->call);
  }
  iree_string_builder_deinitialize(&name_builder);

  if (iree_status_is_ok(status)) {
    status = parse_inputs_into_call(
        &call_state->call,
        iree_runtime_session_device_allocator(program_state->session),
        iree_make_cstring_view(inputs));
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(iree_vm_async_invoke_state_t),
                                   (void**)&(call_state->invoke_state));
  }

  if (iree_status_is_ok(status)) {
    iree_loop_t loop = iree_loop_emscripten(program_state->sample_state->loop);
    iree_vm_context_t* vm_context =
        iree_runtime_session_context(call_state->call.session);
    iree_vm_function_t vm_function = call_state->call.function;
    iree_vm_list_t* inputs = call_state->call.inputs;
    iree_vm_list_t* outputs = call_state->call.outputs;

    // Note: Timing has ~millisecond precision on the web to mitigate timing /
    // side-channel security threats.
    // https://developer.mozilla.org/en-US/docs/Web/API/Performance/now#reduced_time_precision
    call_state->invoke_start_time = iree_time_now();

    status = iree_vm_async_invoke(
        loop, call_state->invoke_state, vm_context, vm_function,
        IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL, inputs, outputs,
        iree_allocator_system(), invoke_callback, /*user_data=*/call_state);
  }

  fprintf(stderr, "after iree_vm_async_invoke\n");

  // TODO(scotttodd): record end time in async callback instead of here
  // TODO(scotttodd): print outputs in async callback instead of here

  // iree_time_t end_time = iree_time_now();
  // iree_time_t time_elapsed = end_time - start_time;

  // iree_string_builder_t outputs_builder;
  // iree_string_builder_initialize(iree_allocator_system(), &outputs_builder);

  // Output a JSON object as a string:
  // {
  //   "total_invoke_time_ms": [number],
  //   "outputs": [semicolon delimited list of formatted outputs]
  // }
  // if (iree_status_is_ok(status)) {
  //   status = iree_string_builder_append_format(
  //       &outputs_builder,
  //       "{ \"total_invoke_time_ms\": %" PRId64 ", \"outputs\": \"",
  //       time_elapsed / 1000000);
  // }
  // if (iree_status_is_ok(status)) {
  //   status = print_outputs_from_call(&call, &outputs_builder);
  // }
  // if (iree_status_is_ok(status)) {
  //   status = iree_string_builder_append_cstring(&outputs_builder, "\"}");
  // }

  if (!iree_status_is_ok(status)) {
    // iree_string_builder_deinitialize(&outputs_builder);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_call_function_state_destroy(call_state);
    return false;
  }

  return true;

  // // Note: this leaks the buffer. It's up to the caller to free it after use.
  // char* outputs_string =
  // strdup(iree_string_builder_buffer(&outputs_builder));
  // iree_string_builder_deinitialize(&outputs_builder);
  // return outputs_string;
}
