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

#ifndef IREE_HAL_VULKAN_VULKAN_DRIVER_H_
#define IREE_HAL_VULKAN_VULKAN_DRIVER_H_

// clang-format off: Must be included before all other headers:
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include <memory>
#include <vector>

#include "iree/hal/cc/driver.h"
#include "iree/hal/vulkan/debug_reporter.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/vulkan_device.h"

namespace iree {
namespace hal {
namespace vulkan {

class VulkanDriver final : public Driver {
 public:
  // Creates a VulkanDriver that manages its own VkInstance.
  static StatusOr<ref_ptr<VulkanDriver>> Create(
      const iree_hal_vulkan_driver_options_t* options,
      ref_ptr<DynamicSymbols> syms);

  // Creates a VulkanDriver that shares an externally managed VkInstance.
  //
  // |options| are checked for compatibility.
  //
  // |syms| must at least have |vkGetInstanceProcAddr| set. Other symbols will
  // be loaded as needed from |instance|.
  //
  // |instance| must remain valid for the life of the returned VulkanDriver.
  static StatusOr<ref_ptr<VulkanDriver>> CreateUsingInstance(
      const iree_hal_vulkan_driver_options_t* options,
      ref_ptr<DynamicSymbols> syms, VkInstance instance);

  ~VulkanDriver() override;

  const ref_ptr<DynamicSymbols>& syms() const { return syms_; }

  VkInstance instance() const { return instance_; }

  StatusOr<std::vector<DeviceInfo>> EnumerateAvailableDevices() override;

  StatusOr<ref_ptr<Device>> CreateDefaultDevice() override;

  StatusOr<ref_ptr<Device>> CreateDevice(
      iree_hal_device_id_t device_id) override;

  // Creates a device that wraps an externally managed VkDevice.
  //
  // The device will schedule commands against the provided queues.
  StatusOr<ref_ptr<Device>> WrapDevice(
      VkPhysicalDevice physical_device, VkDevice logical_device,
      const iree_hal_vulkan_queue_set_t& compute_queue_set,
      const iree_hal_vulkan_queue_set_t& transfer_queue_set);

 private:
  VulkanDriver(ref_ptr<DynamicSymbols> syms, VkInstance instance,
               bool owns_instance,
               iree_hal_vulkan_device_options_t device_options,
               int default_device_index,
               std::unique_ptr<DebugReporter> debug_reporter);

  ref_ptr<DynamicSymbols> syms_;
  VkInstance instance_;
  bool owns_instance_;
  iree_hal_vulkan_device_options_t device_options_;
  int default_device_index_;
  std::unique_ptr<DebugReporter> debug_reporter_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VULKAN_DRIVER_H_
