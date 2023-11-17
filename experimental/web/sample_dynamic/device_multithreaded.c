// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emscripten/threading.h>

#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/hal/local/loaders/system_library_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "iree/task/api.h"

iree_status_t create_device_with_loaders(iree_allocator_t host_allocator,
                                         iree_hal_device_t** out_device) {
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_status_t status = iree_ok_status();

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_plugin_manager_create(
        /*capacity=*/0, host_allocator, &plugin_manager);
  }

  iree_hal_executable_loader_t* loaders[2] = {NULL, NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_library_loader_create(
        plugin_manager, host_allocator, &loaders[loader_count++]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_module_loader_create_isolated(
        /*user_module_count=*/0, /*user_modules=*/NULL, host_allocator,
        &loaders[loader_count++]);
  }

  // Create a task executor.
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_local_memory_size = 0;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_task_topology_initialize_from_group_count(
      /*group_count=*/4, &topology);
  // Note: threads increase memory usage. If using a high thread count, consider
  // passing in a larger WebAssembly.Memory object, increasing Emscripten's
  // INITIAL_MEMORY, or setting Emscripten's ALLOW_MEMORY_GROWTH.
  // iree_task_topology_initialize_from_group_count(
  //     /*group_count=*/emscripten_num_logical_cores(), &topology);
  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create(options, &topology, host_allocator,
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
        identifier, &params, /*queue_count=*/1, &executor, loader_count,
        loaders, device_allocator, host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return status;
}
