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

#include "iree/hal/vulkan/pipeline_cache.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

static const iree_hal_executable_format_t kExecutableFormatSpirV =
    iree_hal_make_executable_format("SPVE");

PipelineCache::PipelineCache(ref_ptr<VkDeviceHandle> logical_device)
    : logical_device_(std::move(logical_device)) {}

PipelineCache::~PipelineCache() = default;

bool PipelineCache::CanPrepareFormat(
    iree_hal_executable_format_t format) const {
  return format == kExecutableFormatSpirV;
}

StatusOr<ref_ptr<Executable>> PipelineCache::PrepareExecutable(
    ExecutableLayout* executable_layout,
    iree_hal_executable_caching_mode_t mode,
    iree_const_byte_span_t executable_data) {
  IREE_TRACE_SCOPE0("PipelineCache::PrepareExecutable");

  // Create the executable (which may itself own many pipelines).
  IREE_ASSIGN_OR_RETURN(
      auto executable,
      PipelineExecutable::Create(
          add_ref(logical_device_),
          /*pipeline_cache=*/VK_NULL_HANDLE,
          static_cast<PipelineExecutableLayout*>(executable_layout), mode,
          executable_data));
  return executable;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
