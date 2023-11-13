// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef INPUT_TOSAINPUTCONVERSION_PASSDETAIL_H_
#define INPUT_TOSAINPUTCONVERSION_PASSDETAIL_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

#define GEN_PASS_CLASSES
#include "input_tosa/InputConversion/Passes.h.inc"

} // namespace iree_compiler
} // namespace mlir

#endif // INPUT_TOSAINPUTCONVERSION_PASSDETAIL_H_
