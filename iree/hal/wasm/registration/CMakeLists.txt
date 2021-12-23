# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::hal
    iree::hal::local::loaders::wasm_module_loader
    iree::hal::local::task_driver
  DEFINES
    "IREE_HAL_HAVE_WASM_DRIVER_MODULE=1"
  PUBLIC
)
