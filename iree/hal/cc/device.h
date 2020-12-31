// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_CC_DEVICE_H_
#define IREE_HAL_CC_DEVICE_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/cc/resource.h"

#if defined(IREE_PLATFORM_WINDOWS)
// Win32 macro name conflicts:
#undef CreateEvent
#undef CreateSemaphore
#endif  // IREE_PLATFORM_WINDOWS

namespace iree {
namespace hal {

class DeviceBase : public ResourceBase<DeviceBase> {
 public:
  virtual ~DeviceBase() { iree_hal_allocator_release(base_.device_allocator); }

  iree_hal_device_t* base() noexcept { return &base_; }

  virtual absl::string_view id() const = 0;

  virtual Status CreateCommandBuffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_command_buffer_t** out_command_buffer) = 0;

  virtual Status CreateDescriptorSet(
      iree_hal_descriptor_set_layout_t* set_layout,
      absl::Span<const iree_hal_descriptor_set_binding_t> bindings,
      iree_hal_descriptor_set_t** out_descriptor_set) = 0;

  virtual Status CreateDescriptorSetLayout(
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings,
      iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) = 0;

  virtual Status CreateEvent(iree_hal_event_t** out_event) = 0;

  virtual Status CreateExecutableCache(
      iree_string_view_t identifier,
      iree_hal_executable_cache_t** out_executable_cache) = 0;

  virtual Status CreateExecutableLayout(
      absl::Span<iree_hal_descriptor_set_layout_t*> set_layouts,
      size_t push_constants,
      iree_hal_executable_layout_t** out_executable_layout) = 0;

  virtual Status CreateSemaphore(uint64_t initial_value,
                                 iree_hal_semaphore_t** out_semaphore) = 0;

  virtual Status QueueSubmit(iree_hal_command_category_t command_categories,
                             uint64_t queue_affinity,
                             iree_host_size_t batch_count,
                             const iree_hal_submission_batch_t* batches) = 0;

  virtual Status WaitSemaphores(iree_hal_wait_mode_t wait_mode,
                                const iree_hal_semaphore_list_t* semaphore_list,
                                iree_time_t deadline_ns) = 0;

  virtual Status WaitIdle(iree_time_t deadline_ns) = 0;

 protected:
  DeviceBase(iree_allocator_t host_allocator,
             iree_hal_allocator_t* device_allocator) {
    static const iree_hal_device_vtable_t vtable = {
        DestroyThunk,
        IdThunk,
        CreateCommandBufferThunk,
        CreateDescriptorSetThunk,
        CreateDescriptorSetLayoutThunk,
        CreateEventThunk,
        CreateExecutableCacheThunk,
        CreateExecutableLayoutThunk,
        CreateSemaphoreThunk,
        QueueSubmitThunk,
        WaitSemaphoresWithDeadlineThunk,
        WaitSemaphoresWithTimeoutThunk,
        WaitIdleWithDeadlineThunk,
        WaitIdleWithTimeoutThunk,
    };
    iree_hal_resource_initialize(&vtable, &base_.resource);
    base_.host_allocator = host_allocator;
    base_.device_allocator = device_allocator;
    iree_hal_allocator_retain(device_allocator);
  }

 private:
  static void DestroyThunk(iree_hal_device_t* device) {
    iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
    reinterpret_cast<DeviceBase*>(device)->~DeviceBase();
    iree_allocator_free(host_allocator, device);
  }

  static iree_string_view_t IdThunk(iree_hal_device_t* device) {
    auto id = reinterpret_cast<DeviceBase*>(device)->id();
    return iree_make_string_view(id.data(), id.size());
  }

  static iree_status_t CreateCommandBufferThunk(
      iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_command_buffer_t** out_command_buffer) {
    //
  }

  static iree_status_t CreateDescriptorSetThunk(
      iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_binding_t* bindings,
      iree_hal_descriptor_set_t** out_descriptor_set) {
    //
  }

  static iree_status_t CreateDescriptorSetLayoutThunk(
      iree_hal_device_t* device,
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_layout_binding_t* bindings,
      iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
    //
  }

  static iree_status_t CreateEventThunk(iree_hal_device_t* device,
                                        iree_hal_event_t** out_event) {
    //
  }

  static iree_status_t CreateExecutableCacheThunk(
      iree_hal_device_t* device, iree_string_view_t identifier,
      iree_hal_executable_cache_t** out_executable_cache) {
    //
  }

  static iree_status_t CreateExecutableLayoutThunk(
      iree_hal_device_t* device, iree_host_size_t set_layout_count,
      iree_hal_descriptor_set_layout_t** set_layouts,
      iree_host_size_t push_constants,
      iree_hal_executable_layout_t** out_executable_layout) {
    //
  }

  static iree_status_t CreateSemaphoreThunk(
      iree_hal_device_t* device, uint64_t initial_value,
      iree_hal_semaphore_t** out_semaphore) {
    //
  }

  static iree_status_t QueueSubmitThunk(
      iree_hal_device_t* device, iree_hal_command_category_t command_categories,
      uint64_t queue_affinity, iree_host_size_t batch_count,
      const iree_hal_submission_batch_t* batches) {
    //
  }

  static iree_status_t WaitSemaphoresWithDeadlineThunk(
      iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
      const iree_hal_semaphore_list_t* semaphore_list,
      iree_time_t deadline_ns) {
    //
  }

  static iree_status_t WaitSemaphoresWithTimeoutThunk(
      iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
      const iree_hal_semaphore_list_t* semaphore_list,
      iree_duration_t timeout_ns) {
    //
  }

  static iree_status_t WaitIdleWithDeadlineThunk(iree_hal_device_t* device,
                                                 iree_time_t deadline_ns) {
    //
  }

  static iree_status_t WaitIdleWithTimeoutThunk(iree_hal_device_t* device,
                                                iree_duration_t timeout_ns) {
    //
  }

  iree_hal_device_t base_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CC_DEVICE_H_
