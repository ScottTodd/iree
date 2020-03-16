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

#include "iree/base/status_matchers.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vulkan_device.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace vulkan {
namespace {

TEST(DynamicSymbolsTest, CreateSemaphore) {
  ASSERT_OK_AND_ASSIGN(auto driver,
                       DriverRegistry::shared_registry()->Create("vulkan"));
  ASSERT_OK_AND_ASSIGN(auto device, driver->CreateDefaultDevice());
  LOG(INFO) << "Created device '" << device->info().name() << "'";
  auto* vulkan_device = reinterpret_cast<VulkanDevice*>(device.get());
  VkDeviceHandle* logical_device = vulkan_device->logical_device();

  VkSemaphoreTypeCreateInfoKHR timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
  timeline_create_info.pNext = nullptr;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
  timeline_create_info.initialValue = 123;

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  VkSemaphore semaphore_handle = VK_NULL_HANDLE;
  VK_CHECK_OK(vulkan_device->syms()->vkCreateSemaphore(
      *logical_device, &create_info, logical_device->allocator(),
      &semaphore_handle));

  VkSemaphoreSignalInfoKHR signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore_handle;
  signal_info.value = 456;

  vulkan_device->syms()->vkSignalSemaphoreKHR(*logical_device, &signal_info);
}

}  // namespace
}  // namespace vulkan
}  // namespace hal
}  // namespace iree
