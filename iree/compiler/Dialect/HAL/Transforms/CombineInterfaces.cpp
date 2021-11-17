// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

// #include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
// #include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
// #include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-hal-combine-interfaces"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

//===----------------------------------------------------------------------===//
// -iree-hal-combine-interfaces
//===----------------------------------------------------------------------===//

class CombineInterfacesPass
    : public PassWrapper<CombineInterfacesPass, OperationPass<ModuleOp>> {
 public:
  CombineInterfacesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-combine-interfaces";
  }

  StringRef getDescription() const override {
    return "TODO(scotttodd): pass description";
  }

  void runOnOperation() override {
    // TODO(scotttodd): implementation
    return;
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCombineInterfacesPass() {
  return std::make_unique<CombineInterfacesPass>();
}

static PassRegistration<CombineInterfacesPass> pass([] {
  return std::make_unique<CombineInterfacesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
