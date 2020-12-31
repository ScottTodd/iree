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

#ifndef IREE_HAL_CC_RESOURCE_H_
#define IREE_HAL_CC_RESOURCE_H_

#include "iree/base/ref_ptr.h"
#include "iree/hal/api.h"

#ifndef __cplusplus
#error "This header is meant for use with C++ HAL implementations."
#endif  // __cplusplus

namespace iree {
namespace hal {

template <typename T>
class ResourceBase {
 public:
  ResourceBase(const ResourceBase&) = delete;
  ResourceBase& operator=(const ResourceBase&) = delete;

  // Adds a reference; used by ref_ptr.
  friend void ref_ptr_add_ref(T* p) {
    volatile iree_atomic_ref_count_t* counter = p->base()->resource.ref_count;
    iree_atomic_ref_count_inc(counter);
  }

  // Releases a reference, potentially deleting the object; used by ref_ptr.
  friend void ref_ptr_release_ref(T* p) {
    volatile iree_atomic_ref_count_t* counter = p->base()->resource.ref_count;
    if (iree_atomic_ref_count_dec(counter) == 1) {
      delete p;
    }
  }

 protected:
  ResourceBase() = default;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CC_RESOURCE_H_
