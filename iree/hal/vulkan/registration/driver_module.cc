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

#include "iree/hal/vulkan/registration/driver_module.h"

#include "absl/flags/flag.h"
#include "iree/base/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/renderdoc_capture_manager.h"
#include "iree/hal/vulkan/vulkan_driver.h"

ABSL_FLAG(bool, vulkan_validation_layers, true,
          "Enables standard Vulkan validation layers.");
ABSL_FLAG(bool, vulkan_debug_utils, true,
          "Enables VK_EXT_debug_utils, records markers, and logs errors.");
ABSL_FLAG(bool, vulkan_debug_report, false,
          "Enables VK_EXT_debug_report and logs errors.");
ABSL_FLAG(bool, vulkan_push_descriptors, true,
          "Enables use of vkCmdPushDescriptorSetKHR, if available.");
ABSL_FLAG(int, vulkan_default_index, 0, "Index of the default Vulkan device.");
ABSL_FLAG(bool, vulkan_renderdoc, false, "Enables RenderDoc API integration.");
ABSL_FLAG(bool, vulkan_force_timeline_semaphore_emulation, false,
          "Uses timeline semaphore emulation even if native support exists.");

// Vulkan Memory Allocator (VMA) flags
#if VMA_RECORDING_ENABLED
ABSL_FLAG(std::string, vma_recording_file, "",
          "File path to write a CSV containing the VMA recording.");
ABSL_FLAG(bool, vma_recording_flush_after_call, false,
          "Flush the VMA recording file after every call (useful if "
          "crashing/not exiting cleanly).");
#endif  // VMA_RECORDING_ENABLED

namespace iree {
namespace hal {
namespace vulkan {
namespace {

StatusOr<ref_ptr<Driver>> CreateVulkanDriver() {
  IREE_TRACE_SCOPE0("CreateVulkanDriver");

  // TODO: validation layers have bugs when using VK_EXT_debug_report, so if the
  // user requested that we force them off with a warning. Prefer using
  // VK_EXT_debug_utils when available.
  if (absl::GetFlag(FLAGS_vulkan_debug_report) &&
      absl::GetFlag(FLAGS_vulkan_validation_layers)) {
    IREE_LOG(WARNING)
        << "VK_EXT_debug_report has issues with modern validation "
           "layers; disabling validation";
    absl::SetFlag(&FLAGS_vulkan_validation_layers, false);
  }

  // Load the Vulkan library. This will fail if the library cannot be found or
  // does not have the expected functions.
  IREE_ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());

  // Setup driver options from flags. We do this here as we want to enable other
  // consumers that may not be using modules/command line flags to be able to
  // set their options however they want.
  iree_hal_vulkan_driver_options_t options;
  iree_hal_vulkan_driver_options_initialize(&options);

  std::vector<const char*> instance_required_layers;
  std::vector<const char*> instance_optional_layers;
  std::vector<const char*> instance_required_extensions;
  std::vector<const char*> instance_optional_extensions;
  std::vector<const char*> device_required_layers;
  std::vector<const char*> device_optional_layers;
  std::vector<const char*> device_required_extensions;
  std::vector<const char*> device_optional_extensions;

  // REQUIRED: these are required extensions that must be present for IREE to
  // work (such as those relied upon by SPIR-V kernels, etc).
  device_required_extensions.push_back(
      VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
  // Multiple extensions depend on VK_KHR_get_physical_device_properties2.
  // This extension was deprecated in Vulkan 1.1 as its functionality was
  // promoted to core, so we list it as optional even though we require it.
  instance_optional_extensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  // Timeline semaphore support is optional and will be emulated if necessary.
  device_optional_extensions.push_back(
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  // Polyfill layer - enable if present (instead of our custom emulation).
  instance_optional_layers.push_back("VK_LAYER_KHRONOS_timeline_semaphore");

  if (absl::GetFlag(FLAGS_vulkan_validation_layers)) {
    instance_optional_layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  if (absl::GetFlag(FLAGS_vulkan_debug_report)) {
    instance_optional_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  }
  if (absl::GetFlag(FLAGS_vulkan_debug_utils)) {
    instance_optional_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (absl::GetFlag(FLAGS_vulkan_push_descriptors)) {
    device_optional_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  options.extensibility_spec.required_layers =
      absl::MakeConstSpan(instance_required_layers);
  options.extensibility_spec.optional_layers =
      absl::MakeConstSpan(instance_optional_layers);
  options.extensibility_spec.required_extensions =
      absl::MakeConstSpan(instance_required_extensions);
  options.extensibility_spec.optional_extensions =
      absl::MakeConstSpan(instance_optional_extensions);
  options.device_options.extensibility_spec.required_layers =
      absl::MakeConstSpan(device_required_layers);
  options.device_options.extensibility_spec.optional_layers =
      absl::MakeConstSpan(device_optional_layers);
  options.device_options.extensibility_spec.required_extensions =
      absl::MakeConstSpan(device_required_extensions);
  options.device_options.extensibility_spec.optional_extensions =
      absl::MakeConstSpan(device_optional_extensions);

  options.default_device_index = absl::GetFlag(FLAGS_vulkan_default_index);
  options.device_options.force_timeline_semaphore_emulation =
      absl::GetFlag(FLAGS_vulkan_force_timeline_semaphore_emulation);

#if VMA_RECORDING_ENABLED
  auto& record_settings = options.device_options.vma_record_settings;
  record_settings.flags = absl::GetFlag(FLAGS_vma_recording_flush_after_call)
                              ? VMA_RECORD_FLUSH_AFTER_CALL_BIT
                              : 0;
  record_settings.pFilePath = absl::GetFlag(FLAGS_vma_recording_file);
#endif  // VMA_RECORDING_ENABLED

  // Create the driver and VkInstance.
  return VulkanDriver::Create(options, std::move(syms));
}

}  // namespace
}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#include <inttypes.h>

#define IREE_HAL_VULKAN_1_X_DRIVER_ID 0x564C4B31u  // VLK1

static iree_status_t iree_hal_vulkan_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  // NOTE: we could query supported vulkan versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      /*driver_id=*/IREE_HAL_VULKAN_1_X_DRIVER_ID,
      /*driver_name=*/iree_make_cstring_view("vulkan"),
      /*full_name=*/iree_make_cstring_view("Vulkan 1.x (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_VULKAN_1_X_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }
  IREE_ASSIGN_OR_RETURN(auto driver, iree::hal::vulkan::CreateVulkanDriver());
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_vulkan_driver_factory_enumerate,
      iree_hal_vulkan_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
