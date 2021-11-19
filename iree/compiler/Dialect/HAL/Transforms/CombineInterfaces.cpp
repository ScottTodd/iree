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
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
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

    // "the materialize interfaces (2) pass adds an attribute to each dispatch
    // with which bindings things map to"
    // "so I think the easiest thing would be to have a pass that runs after
    // materialize interfaces (2) but before lowering into HAL that just updates
    // the interfaces and the binding symbols there ignoring the names, it could
    // just change the hal.interfaces (adjust the set/binding numbers) then no
    // other IR needs to change"

    LLVM_DEBUG({ llvm::dbgs() << "Analyzing interfaces:\n"; });
    getOperation()->walk([&](IREE::HAL::InterfaceOp interfaceOp) {
      // Note: public interface and private interface, both matching
      auto executableOp =
          interfaceOp->getParentOfType<IREE::HAL::ExecutableOp>();
      LLVM_DEBUG({
        if (interfaceOp.isPublic()) {
          llvm::dbgs() << "Executable '" << executableOp.getName()
                       << "' has public interface:\n";
          interfaceOp.dump();
        }
      });
    });

    LLVM_DEBUG({ llvm::dbgs() << "Analyzing dispatches:\n"; });
    getOperation()->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      auto bindingSymbols = dispatchOp->getAttr("hal.interface.bindings")
                                .dyn_cast_or_null<ArrayAttr>();
      LLVM_DEBUG({
        llvm::dbgs() << "Bindings: ";
        for (auto bindingSymbol : bindingSymbols) {
          llvm::dbgs() << bindingSymbol << ", ";
        }
        llvm::dbgs() << "\n";
      });
      // auto bindingOps = llvm::to_vector<
      //     4>(llvm::map_range(bindingSymbols, [&](Attribute symRefAttr) {
      //   auto bindingOp =
      //       SymbolTable::lookupNearestSymbolFrom<IREE::HAL::InterfaceBindingOp>(
      //           dispatchOp, symRefAttr.cast<SymbolRefAttr>());
      //   assert(bindingOp && "binding not found");
      //   return bindingOp;
      // }));
    });

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
