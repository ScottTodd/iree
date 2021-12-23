# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    wasm
  DRIVER_REGISTRATION_HDR
    "iree/hal/wasm/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_wasm_driver_module_register"
  DEPS
    iree::hal::wasm::registration
)
