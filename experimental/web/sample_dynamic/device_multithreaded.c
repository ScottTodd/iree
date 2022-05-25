// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emscripten/threading.h>

#include "iree/hal/local/loaders/system_library_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/task/api.h"

iree_status_t create_device_with_wasm_loader(iree_allocator_t host_allocator,
                                             iree_hal_device_t** out_device) {
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_status_t status = iree_ok_status();

  iree_hal_executable_loader_t* loaders[1] = {NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_library_loader_create(
        iree_hal_executable_import_provider_null(), host_allocator,
        &loaders[loader_count++]);
  }

  // Create a task executor.
  iree_task_executor_t* executor = NULL;
  iree_task_scheduling_mode_t scheduling_mode = 0;
  iree_host_size_t worker_local_memory = 0;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_task_topology_initialize_from_group_count(
      /*group_count=*/2, &topology);
  // Note: threads increase memory usage. If using a high thread count, consider
  // passing in a larger WebAssembly.Memory object, increasing Emscripten's
  // INITIAL_MEMORY, or setting Emscripten's ALLOW_MEMORY_GROWTH.
  // iree_task_topology_initialize_from_group_count(
  //     /*group_count=*/emscripten_num_logical_cores(), &topology);
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create(scheduling_mode, &topology,
                                       worker_local_memory, host_allocator,
                                       &executor);
  }
  iree_task_topology_deinitialize(&topology);

  iree_string_view_t identifier = iree_make_cstring_view("task");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_create(
        identifier, &params, executor, loader_count, loaders, device_allocator,
        host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return status;
}
