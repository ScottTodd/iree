// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_ALLOCATOR_TEST_H_
#define IREE_HAL_CTS_ALLOCATOR_TEST_H_

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

namespace {

constexpr iree_device_size_t kDefaultAllocationSize = 1024;

}  // namespace

class allocator_test : public CtsTestBase {};

// All allocators must support some baseline capabilities.
//
// Certain capabilities or configurations are optional and may vary between
// driver implementations or target devices, such as:
//   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
//   IREE_HAL_BUFFER_USAGE_MAPPING
TEST_P(allocator_test, BaselineBufferCompatibility) {
  // Need at least one way to get data between the host and device.
  //   host local + device visible --(transfer)--> host visible + device local
  iree_hal_buffer_params_t host_local_params = {0};
  host_local_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_local_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t host_local_allocation_size = 0;
  iree_hal_buffer_compatibility_t transfer_compatibility_host =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, host_local_params, kDefaultAllocationSize,
          &host_local_params, &host_local_allocation_size);
  EXPECT_GE(host_local_allocation_size, kDefaultAllocationSize);

  iree_hal_buffer_params_t device_local_params = {0};
  device_local_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  device_local_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t device_allocation_size = 0;
  iree_hal_buffer_compatibility_t transfer_compatibility_device =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, device_local_params, kDefaultAllocationSize,
          &device_local_params, &device_allocation_size);
  EXPECT_GE(device_allocation_size, kDefaultAllocationSize);

  iree_hal_buffer_compatibility_t required_transfer_compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  EXPECT_TRUE(iree_all_bits_set(transfer_compatibility_host,
                                required_transfer_compatibility) ||
              iree_all_bits_set(transfer_compatibility_device,
                                required_transfer_compatibility));

  // Need to be able to use some type of buffer as dispatch inputs or outputs.
  //   host local + device visible --(dispatch storage)-->
  iree_hal_buffer_params_t dispatch_params = {0};
  dispatch_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  dispatch_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  iree_device_size_t dispatch_allocation_size = 0;
  iree_hal_buffer_compatibility_t dispatch_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, dispatch_params, kDefaultAllocationSize,
          &dispatch_params, &dispatch_allocation_size);
  EXPECT_GE(dispatch_allocation_size, kDefaultAllocationSize);
  EXPECT_TRUE(
      iree_all_bits_set(dispatch_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_P(allocator_test, AllocateInputBuffers) {
  std::vector<iree_device_size_t> test_sizes = {
      0,     // Corner case
      4,     // Small power of two
      8,     // Small power of two
      64,    // Small power of two
      100,   // Non-power of two
      1024,  // Medium power of two
  };

  for (auto test_size : test_sizes) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      test_size, &buffer));

    // At a mimimum, the requested memory type should be respected.
    // Additional bits may be optionally set depending on the allocator.
    EXPECT_TRUE(
        iree_all_bits_set(iree_hal_buffer_memory_type(buffer), params.type));
    EXPECT_TRUE(
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), params.usage));
    // Larger allocation sizes are okay.
    EXPECT_GE(iree_hal_buffer_allocation_size(buffer), test_size);

    iree_hal_buffer_release(buffer);
  }
}

TEST_P(allocator_test, AllocateOutputBuffers) {
  std::vector<iree_device_size_t> test_sizes = {
      0,     // Corner case
      4,     // Small power of two
      8,     // Small power of two
      64,    // Small power of two
      100,   // Non-power of two
      1024,  // Medium power of two
  };

  for (auto test_size : test_sizes) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    iree_hal_buffer_t* buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      test_size, &buffer));

    // At a mimimum, the requested memory type should be respected.
    // Additional bits may be optionally set depending on the allocator.
    EXPECT_TRUE(
        iree_all_bits_set(iree_hal_buffer_memory_type(buffer), params.type));
    EXPECT_TRUE(
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), params.usage));
    // Larger allocation sizes are okay.
    EXPECT_GE(iree_hal_buffer_allocation_size(buffer), test_size);

    iree_hal_buffer_release(buffer);
  }
}

// TODO(scotttodd): test ExportBuffer and ImportBuffer (somehow)
//   maybe with GTEST_SKIP if unavailable/unimplemented

// TEST_P(allocator_test, ExportBuffer) {
//   iree_hal_buffer_params_t params = {0};
//   params.type =
//       IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
//   params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
//   iree_hal_buffer_t* buffer = NULL;
//   IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_,
//   params,
//                                                     kDefaultAllocationSize,
//                                                     &buffer));

//   iree_hal_external_buffer_t exported_buffer;
//   IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
//       device_allocator_, buffer,
//       IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
//       IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));

//   EXPECT_TRUE(iree_all_bits_set(IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
//                                 exported_buffer.type));
//   EXPECT_GE(exported_buffer.size, kDefaultAllocationSize);  // Larger is
//   okay.

//   iree_hal_buffer_release(buffer);
// }

// TEST_P(allocator_test, ImportBuffer) {
//   iree_hal_buffer_params_t params = {0};
//   params.type =
//       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
//       IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
//   params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
//   iree_hal_buffer_t* buffer = NULL;
//   IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
//       device_allocator_, params, /*allocation_size=*/0, &buffer));
// }

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_ALLOCATOR_TEST_H_
