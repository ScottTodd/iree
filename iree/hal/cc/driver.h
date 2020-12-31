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

#ifndef IREE_HAL_CC_DRIVER_H_
#define IREE_HAL_CC_DRIVER_H_

#include <memory>
#include <string>
#include <vector>

#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/cc/device.h"
#include "iree/hal/cc/device_info.h"

namespace iree {
namespace hal {

class Driver : public RefObject<Driver> {
 public:
  virtual ~Driver() = default;

  virtual StatusOr<std::vector<DeviceInfo>> EnumerateAvailableDevices() = 0;

  virtual StatusOr<ref_ptr<Device>> CreateDefaultDevice() = 0;

  virtual StatusOr<ref_ptr<Device>> CreateDevice(
      iree_hal_device_id_t device_id) = 0;
  StatusOr<ref_ptr<Device>> CreateDevice(const DeviceInfo& device_info) {
    return CreateDevice(device_info.device_id());
  }

 protected:
  explicit Driver(std::string name) : name_(std::move(name)) {}

 private:
  const std::string name_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CC_DRIVER_H_
