// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/task_command_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/device.h"
#include "iree/hal/local/local_descriptor_set_layout.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_executable_layout.h"
#include "iree/task/list.h"
#include "iree/task/task.h"

// iree/task/-based command buffer.
// We track a minimal amount of state here and incrementally build out the task
// DAG that we can submit to the task system directly. There's no intermediate
// data structures and we produce the iree_task_ts directly. In the steady state
// all allocations are served from a shared per-device block pool with no
// additional allocations required during recording or execution. That means our
// command buffer here is essentially just a builder for the task system types
// and manager of the lifetime of the tasks.
typedef struct {
  iree_hal_resource_t resource;

  iree_hal_device_t* device;
  iree_task_scope_t* scope;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;

  // Arena used for all allocations; references the shared device block pool.
  iree_arena_allocator_t arena;

  // One or more tasks at the root of the command buffer task DAG.
  // These tasks are all able to execute concurrently and will be the initial
  // ready task set in the submission.
  iree_task_list_t root_tasks;

  // One or more tasks at the leaves of the DAG.
  // Only once all these tasks have completed execution will the command buffer
  // be considered completed as a whole.
  //
  // An empty list indicates that root_tasks are also the leaves.
  iree_task_list_t leaf_tasks;

  // State tracked within the command buffer during recording only.
  struct {
    // The last global barrier that was inserted, if any.
    // The barrier is allocated and inserted into the DAG when requested but the
    // actual barrier dependency list is only allocated and set on flushes.
    // This lets us allocate the appropriately sized barrier task list from the
    // arena even though when the barrier is recorded we don't yet know what
    // other tasks we'll be emitting as we walk the command stream.
    iree_task_barrier_t* open_barrier;

    // A flattened list of all available descriptor set bindings.
    // As descriptor sets are pushed/bound the bindings will be updated to
    // represent the fully-translated binding data pointer.
    // TODO(benvanik): support proper mapping semantics and track the
    // iree_hal_buffer_mapping_t and map/unmap where appropriate.
    iree_hal_executable_binding_ptr_t
        bindings[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                 IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];
    iree_device_size_t
        binding_lengths[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                        IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];

    // All available push constants updated each time push_constants is called.
    // Reset only with the command buffer and otherwise will maintain its values
    // during recording to allow for partial push_constants updates.
    uint32_t* push_constants[IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT];
  } state;
} iree_hal_task_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_task_command_buffer_vtable;

iree_status_t iree_hal_task_command_buffer_create(
    iree_hal_device_t* device, iree_task_scope_t* scope,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_arena_block_pool_t* block_pool,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  if (mode != IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) {
    // If we want reuse we'd need to support duplicating the task DAG after
    // recording or have some kind of copy-on-submit behavior that does so if
    // a command buffer is submitted for execution twice. Allowing for the same
    // command buffer to be enqueued multiple times would be fine so long as
    // execution doesn't overlap (`cmdbuf|cmdbuf` vs
    // `cmdbuf -> semaphore -> cmdbuf`) though we'd still need to be careful
    // that we did the enqueuing and reset of the task structures at the right
    // times. Definitely something that'll be useful in the future... but not
    // today :)
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only one-shot command buffer usage is supported");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_device_host_allocator(device),
                            sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_task_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->device = device;
    command_buffer->scope = scope;
    command_buffer->mode = mode;
    command_buffer->allowed_categories = command_categories;
    iree_arena_initialize(block_pool, &command_buffer->arena);
    iree_task_list_initialize(&command_buffer->root_tasks);
    memset(&command_buffer->state, 0, sizeof(command_buffer->state));
    iree_task_list_initialize(&command_buffer->leaf_tasks);
    *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_command_buffer_reset(
    iree_hal_task_command_buffer_t* command_buffer) {
  iree_task_list_discard(&command_buffer->leaf_tasks);
  memset(&command_buffer->state, 0, sizeof(command_buffer->state));
  iree_task_list_discard(&command_buffer->root_tasks);
  iree_arena_reset(&command_buffer->arena);
}

static void iree_hal_task_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(command_buffer->device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_command_buffer_reset(command_buffer);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_command_category_t
iree_hal_task_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;
  return command_buffer->allowed_categories;
}

//===----------------------------------------------------------------------===//
// iree_hal_task_command_buffer_t state management utilities
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_task_command_buffer_stitch_batch(
    const iree_hal_submission_batch_t* batch,
    iree_task_submission_t* out_submission) {
  // DO NOT SUBMIT
}

// DO NOT SUBMIT emit_global_barrier
// allocate barrier
// for each open task:
//   set completion task to new barrier
// if prev_barrier:
//   if 1 task then set =0 and use completion_task
// else no prev_barrier:
//   move open tasks to root tasks
// if no root tasks set to barrier

static iree_status_t iree_hal_task_command_buffer_flush_tasks(
    iree_hal_task_command_buffer_t* command_buffer) {
  // state.open_barrier
}

static iree_status_t iree_hal_task_command_buffer_emit_global_barrier(
    iree_hal_task_command_buffer_t* command_buffer) {
  // flush (set open_barrier = null)
  // alloc barrier
  // open_barrier = new barrier
}

static iree_status_t iree_hal_task_command_buffer_emit_execution_task(
    iree_hal_task_command_buffer_t* command_buffer, iree_task_t* task) {
  iree_task_list_push_back(&command_buffer->leaf_tasks, task);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_begin/end
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;
  iree_hal_task_command_buffer_reset(command_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_task_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;
  return iree_hal_task_command_buffer_flush_tasks(command_buffer);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execution_barrier
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  // TODO(benvanik): actual DAG construction. Right now we are just doing simple
  // global barriers each time and forcing a join-fork point.
  return iree_hal_task_command_buffer_emit_global_barrier(command_buffer);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_signal_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "task events not yet implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_reset_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "task events not yet implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_wait_events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "task events not yet implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_discard_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//
// NOTE: for large fills we could dispatch this as tiles for parallelism.
// We'd want to do some measurement for when it's worth it; filling a 200KB
// buffer: maybe not, filling a 200MB buffer: yeah.

typedef struct {
  iree_task_call_t task;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint32_t pattern_length;
  uint8_t pattern[8];
} iree_hal_cmd_fill_buffer_t;

static iree_status_t iree_hal_cmd_fill_buffer(uintptr_t user_context,
                                              uintptr_t task_context) {
  const iree_hal_cmd_fill_buffer_t* cmd =
      (const iree_hal_cmd_fill_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_buffer_fill(cmd->target_buffer, cmd->target_offset, cmd->length,
                           cmd->pattern, cmd->pattern_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  iree_hal_cmd_fill_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&command_buffer->arena, sizeof(*cmd), (void**)&cmd));

  iree_task_call_initialize(
      command_buffer->scope,
      iree_task_make_closure(iree_hal_cmd_fill_buffer, (uintptr_t)cmd),
      &cmd->task);
  cmd->target_buffer = target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;
  memcpy(cmd->pattern, pattern, pattern_length);
  cmd->pattern_length = pattern_length;

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

typedef struct {
  iree_task_call_t task;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint8_t source_buffer[];
} iree_hal_cmd_update_buffer_t;

static iree_status_t iree_hal_cmd_update_buffer(uintptr_t user_context,
                                                uintptr_t task_context) {
  const iree_hal_cmd_update_buffer_t* cmd =
      (const iree_hal_cmd_update_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_write_data(
      cmd->target_buffer, cmd->target_offset, cmd->source_buffer, cmd->length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  iree_host_size_t total_cmd_size =
      sizeof(iree_hal_cmd_update_buffer_t) + length;

  iree_hal_cmd_update_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           total_cmd_size, (void**)&cmd));

  iree_task_call_initialize(
      command_buffer->scope,
      iree_task_make_closure(iree_hal_cmd_update_buffer, (uintptr_t)cmd),
      &cmd->task);
  cmd->target_buffer = (iree_hal_buffer_t*)target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;

  memcpy(cmd->source_buffer, (const uint8_t*)source_buffer + source_offset,
         cmd->length);

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//
// NOTE: for large copies we could dispatch this as tiles for parallelism.
// We'd want to do some measurement for when it's worth it; copying a 200KB
// buffer: maybe not, copying a 200MB buffer: yeah.

typedef struct {
  iree_task_call_t task;
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_cmd_copy_buffer_t;

static iree_status_t iree_hal_cmd_copy_buffer(uintptr_t user_context,
                                              uintptr_t task_context) {
  const iree_hal_cmd_copy_buffer_t* cmd =
      (const iree_hal_cmd_copy_buffer_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_copy_data(
      cmd->source_buffer, cmd->source_offset, cmd->target_buffer,
      cmd->target_offset, cmd->length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  iree_hal_cmd_copy_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&command_buffer->arena, sizeof(*cmd), (void**)&cmd));

  iree_task_call_initialize(
      command_buffer->scope,
      iree_task_make_closure(iree_hal_cmd_copy_buffer, (uintptr_t)cmd),
      &cmd->task);
  cmd->source_buffer = (iree_hal_buffer_t*)source_buffer;
  cmd->source_offset = source_offset;
  cmd->target_buffer = (iree_hal_buffer_t*)target_buffer;
  cmd->target_offset = target_offset;
  cmd->length = length;

  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_constants
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_task_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  if (IREE_UNLIKELY(offset + values_length >=
                    IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range %zu (length=%zu) out of range",
                            offset, values_length);
  }

  memcpy(&command_buffer->state.push_constants[offset], values,
         values_length * sizeof(uint32_t));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_descriptor_set
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_task_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  if (IREE_UNLIKELY(set >= IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set %zu out of bounds", set);
  }

  iree_hal_local_executable_layout_t* local_executable_layout =
      iree_hal_local_executable_layout_cast(executable_layout);
  iree_hal_local_descriptor_set_layout_t* local_set_layout =
      iree_hal_local_descriptor_set_layout_cast(
          local_executable_layout->set_layouts[set]);

  iree_host_size_t binding_base =
      set * IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (IREE_UNLIKELY(bindings[i].binding >=
                      IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer binding index out of bounds");
    }
    iree_host_size_t binding_ordinal = binding_base + bindings[i].binding;

    // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
    iree_hal_buffer_mapping_t buffer_mapping;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        bindings[i].buffer, local_set_layout->bindings[binding_ordinal].access,
        bindings[i].offset, bindings[i].length, &buffer_mapping));
    command_buffer->state.bindings[binding_ordinal] =
        buffer_mapping.contents.data;
    command_buffer->state.binding_lengths[binding_ordinal] =
        buffer_mapping.contents.data_length;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_bind_descriptor_set
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_task_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "descriptor set binding not yet implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

typedef struct {
  iree_task_dispatch_t task;
  iree_hal_local_executable_t* executable;
  iree_host_size_t ordinal;
  iree_hal_executable_binding_ptr_t* IREE_RESTRICT bindings;
  iree_device_size_t* IREE_RESTRICT binding_lengths;
  uint32_t* IREE_RESTRICT push_constants;
} iree_hal_cmd_dispatch_t;

static iree_status_t iree_hal_cmd_dispatch_tile(uintptr_t user_context,
                                                uintptr_t task_context) {
  const iree_hal_cmd_dispatch_t* cmd =
      (const iree_hal_cmd_dispatch_t*)user_context;
  const iree_task_tile_context_t* tile_context =
      (const iree_task_tile_context_t*)task_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_dispatch_state_v0_t state;
  // TODO(benvanik): wire up device state (imports, etc) and cache on the
  // command buffer for reuse across all tiles.

  iree_hal_local_executable_call_t call = {
      .state = &state,
      .push_constants = cmd->push_constants,
      .bindings = cmd->bindings,
  };
  memcpy(call.workgroup_id.value, tile_context->workgroup_xyz,
         sizeof(iree_hal_vec3_t));
  memcpy(call.workgroup_size.value, tile_context->workgroup_size,
         sizeof(iree_hal_vec3_t));
  memcpy(call.workgroup_count.value, tile_context->workgroup_count,
         sizeof(iree_hal_vec3_t));
  iree_status_t status = iree_hal_local_executable_issue_call(
      cmd->executable, cmd->ordinal, &call);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_command_buffer_build_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z,
    iree_hal_cmd_dispatch_t** out_cmd) {
  iree_hal_task_command_buffer_t* command_buffer =
      (iree_hal_task_command_buffer_t*)base_command_buffer;

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  iree_host_size_t push_constant_count =
      local_executable->layout->push_constants;
  iree_hal_local_binding_mask_t used_binding_mask =
      local_executable->layout->used_bindings;
  iree_host_size_t used_binding_count =
      iree_math_count_ones_u64(used_binding_mask);

  iree_host_size_t total_cmd_size =
      sizeof(iree_hal_cmd_dispatch_t) + push_constant_count * sizeof(uint32_t) +
      used_binding_count * sizeof(iree_hal_executable_binding_ptr_t) +
      used_binding_count * sizeof(iree_device_size_t);

  iree_hal_cmd_dispatch_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           total_cmd_size, (void**)&cmd));

  cmd->executable = local_executable;
  cmd->ordinal = entry_point;

  uint32_t workgroup_count[3] = {workgroup_x, workgroup_y, workgroup_z};
  // TODO(benvanik): expose on API or keep fixed on executable.
  uint32_t workgroup_size[3] = {1, 1, 1};
  iree_task_dispatch_initialize(
      command_buffer->scope,
      iree_task_make_closure(iree_hal_cmd_dispatch_tile, (uintptr_t)cmd),
      workgroup_size, workgroup_count, &cmd->task);

  // Copy only the push constant range used by the executable.
  cmd->push_constants = (uint32_t*)((uint8_t*)cmd + sizeof(*cmd));
  memcpy(cmd->push_constants, command_buffer->state.push_constants,
         push_constant_count * sizeof(*cmd->push_constants));

  // Produce the dense binding list based on the declared bindings used.
  // This allows us to change the descriptor sets and bindings counts supported
  // in the HAL independent of any executable as each executable just gets the
  // flat dense list and doesn't care about our descriptor set stuff.
  //
  // Note that we are just directly setting the binding data pointers here with
  // no ownership/retaining/etc - it's part of the HAL contract that buffers are
  // kept valid for the duration they may be in use.
  cmd->bindings =
      (iree_hal_executable_binding_ptr_t*)((uint8_t*)cmd->push_constants +
                                           push_constant_count *
                                               sizeof(*cmd->push_constants));
  cmd->binding_lengths =
      (iree_device_size_t*)((uint8_t*)cmd->bindings +
                            used_binding_count * sizeof(*cmd->bindings));
  iree_host_size_t binding_base = 0;
  for (iree_host_size_t i = 0; i < used_binding_count; ++i) {
    int mask_offset = iree_math_count_trailing_zeros_u64(used_binding_mask);
    int binding_ordinal = binding_base + mask_offset;
    binding_base += mask_offset + 1;
    used_binding_mask = used_binding_mask >> (mask_offset + 1);
    cmd->bindings[i] = command_buffer->state.bindings[binding_ordinal];
    cmd->binding_lengths[i] =
        command_buffer->state.binding_lengths[binding_ordinal];
  }

  *out_cmd = cmd;
  return iree_hal_task_command_buffer_emit_execution_task(command_buffer,
                                                          &cmd->task.header);
}

static iree_status_t iree_hal_task_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cmd_dispatch_t* cmd = NULL;
  return iree_hal_task_command_buffer_build_dispatch(
      base_command_buffer, executable, entry_point, workgroup_x, workgroup_y,
      workgroup_z, &cmd);
}

static iree_status_t iree_hal_task_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
  iree_hal_buffer_mapping_t buffer_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      workgroups_buffer, IREE_HAL_MEMORY_ACCESS_READ, workgroups_offset,
      3 * sizeof(uint32_t), &buffer_mapping));

  iree_hal_cmd_dispatch_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_command_buffer_build_dispatch(
      base_command_buffer, executable, entry_point, 0, 0, 0, &cmd));
  cmd->task.workgroup_count.ptr = (const uint32_t*)buffer_mapping.contents.data;
  cmd->task.header.flags |= IREE_TASK_FLAG_DISPATCH_INDIRECT;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_vtable_t
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_task_command_buffer_vtable = {
        .destroy = iree_hal_task_command_buffer_destroy,
        .allowed_categories = iree_hal_task_command_buffer_allowed_categories,
        .begin = iree_hal_task_command_buffer_begin,
        .end = iree_hal_task_command_buffer_end,
        .execution_barrier = iree_hal_task_command_buffer_execution_barrier,
        .signal_event = iree_hal_task_command_buffer_signal_event,
        .reset_event = iree_hal_task_command_buffer_reset_event,
        .wait_events = iree_hal_task_command_buffer_wait_events,
        .discard_buffer = iree_hal_task_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_task_command_buffer_fill_buffer,
        .update_buffer = iree_hal_task_command_buffer_update_buffer,
        .copy_buffer = iree_hal_task_command_buffer_copy_buffer,
        .push_constants = iree_hal_task_command_buffer_push_constants,
        .push_descriptor_set = iree_hal_task_command_buffer_push_descriptor_set,
        .bind_descriptor_set = iree_hal_task_command_buffer_bind_descriptor_set,
        .dispatch = iree_hal_task_command_buffer_dispatch,
        .dispatch_indirect = iree_hal_task_command_buffer_dispatch_indirect,
};
