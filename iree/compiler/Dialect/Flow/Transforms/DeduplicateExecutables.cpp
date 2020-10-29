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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
// #include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

// TODO(scotttodd): pass statistics (number of executables deduped)
// https://mlir.llvm.org/docs/PassManagement/#pass-statistics

#include <iostream>  // DO NOT SUBMIT

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// bool InterfaceOp::isEquivalentTo(InterfaceOp other) {
//   auto bindings =
//   llvm::to_vector<4>(getBlock().getOps<InterfaceBindingOp>()); auto
//   otherBindings =
//       llvm::to_vector<4>(other.getBlock().getOps<InterfaceBindingOp>());
//   return bindings.size() == otherBindings.size() &&
//          llvm::all_of(llvm::zip(bindings, otherBindings), [](auto bindings) {
//            return OperationEquivalence::isEquivalentTo(std::get<0>(bindings),
//                                                        std::get<1>(bindings));
//          });
// }

bool areExecutablesEquivalent(ExecutableOp lhs, ExecutableOp rhs) {
  // std::cerr << "LHS: " << std::endl;
  // lhs.dump();
  // std::cerr << "RHS: " << std::endl;
  // rhs.dump();
  auto lhsModule = lhs.getInnerModule();
  auto rhsModule = rhs.getInnerModule();
  return OperationEquivalence::isEquivalentTo(lhsModule, rhsModule);
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
void replaceEntryPointUses(mlir::ModuleOp moduleOp,
                           const DenseMap<Attribute, Attribute> &replacements) {
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    funcOp.walk([&](DispatchOp dispatchOp) {
      auto it = replacements.find(dispatchOp.entry_point());
      if (it != replacements.end()) {
        dispatchOp.entry_pointAttr(it->second.cast<SymbolRefAttr>());
      }
    });
  }
}

}  // namespace

class DeduplicateExecutablesPass
    : public PassWrapper<DeduplicateExecutablesPass, OperationPass<ModuleOp>> {
 public:
  DeduplicateExecutablesPass() = default;

  void runOnOperation() override {
    // optimize:
    //   hashmap <hash, flow executable op>
    //   for each flow executable op:
    //     hash op (how?)
    //     if hash exists in hashmap:
    //       dedup (rewrite references, delete)
    //     else:
    //       add to hashmap

    // brute force:
    //   for each flow executable op:
    //     for each other flow executable op:
    //       if equivalent:
    //         dedup

    auto moduleOp = getOperation();
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    if (executableOps.size() < 2) return;  // DO NOT SUBMIT

    DenseMap<Attribute, Attribute> entryPointRefReplacements;

    auto executable0 = executableOps[0];
    auto executable1 = executableOps[1];
    if (areExecutablesEquivalent(executable0, executable1)) {
      std::cerr << "equivalent" << std::endl;

      auto executable0EntryOp = executable0.getDispatchEntryOp();
      auto executable1EntryOp = executable1.getDispatchEntryOp();
      auto oldSymbolRefAttr = builder.getSymbolRefAttr(
          executable1.getName(),
          {builder.getSymbolRefAttr(executable1EntryOp.sym_name())});
      auto newSymbolRefAttr = builder.getSymbolRefAttr(
          executable0.getName(),
          {builder.getSymbolRefAttr(executable0EntryOp.sym_name())});
      entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      // auto executable0EntryOps =
      //     llvm::to_vector<8>(executable0.getOps<DispatchEntryOp>());
      // auto executable1EntryOps =
      //     llvm::to_vector<8>(executable1.getOps<DispatchEntryOp>());
      // // TODO(scotttodd): error check
      // for (int i = 0; i < executable0EntryOps.size(); ++i) {
      //   auto oldSymbolRefAttr = builder.getSymbolRefAttr(
      //       executable1.getName(),
      //       {builder.getSymbolRefAttr(executable1EntryOps[i].sym_name())});
      //   auto newSymbolRefAttr = builder.getSymbolRefAttr(
      //       executable0.getName(),
      //       {builder.getSymbolRefAttr(executable0EntryOps[i].sym_name())});
      //   entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      // }

      replaceEntryPointUses(moduleOp, entryPointRefReplacements);

      SymbolTable::replaceAllSymbolUses(executable1.sym_name(),
                                        executable0.sym_name(),
                                        &moduleOp.getBodyRegion());

      executable1.erase();
    } else {
      std::cerr << "not equivalent" << std::endl;
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDeduplicateExecutablesPass() {
  return std::make_unique<DeduplicateExecutablesPass>();
}

static PassRegistration<DeduplicateExecutablesPass> pass(
    "iree-flow-dedupliclate-executables",
    "Deduplicates executables that are identical");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
