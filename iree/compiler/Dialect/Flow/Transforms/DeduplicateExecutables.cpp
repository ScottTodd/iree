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
  auto lhsModule = lhs.getInnerModule();
  auto rhsModule = rhs.getInnerModule();
  // TODO(scotttodd): recurse into modules, check funcOp contents
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
    // if (executableOps.size() < 2) return;  // DO NOT SUBMIT

    SmallVector<ExecutableOp, 3> duplicateExecutableOps;
    DenseMap<Attribute, Attribute> entryPointRefReplacements;

    // For each executable, find the first executable which it is equivalent to.
    // Iteration order:
    //   3 == 0 ? no
    //   3 == 1 ? yes -> mark and move on to 2
    //   2 == 0 ? no
    //   2 == 1 ? yes -> mark and move on to 1
    //   1 == 0 ? no -> not a duplicate, keep it
    //   Done iterating. 0 and 1 stay, 2 and 3 are duplicates of 1.
    for (int i = executableOps.size() - 1; i >= 0; --i) {
      auto possiblyDuplicateExecutable = executableOps[i];
      for (int j = 0; j < i; ++j) {
        std::cerr << "i: " << i << ", j: " << j << std::endl;

        auto comparisonExecutable = executableOps[j];
        if (!areExecutablesEquivalent(possiblyDuplicateExecutable,
                                      comparisonExecutable)) {
          continue;
        }

        std::cerr << "Duplicate! replacing " << i << " with " << j << std::endl;

        // Add to replacement table and break to move to the next possible dup.
        auto oldSymbolRefAttr = builder.getSymbolRefAttr(
            possiblyDuplicateExecutable.getName(),
            {builder.getSymbolRefAttr(
                possiblyDuplicateExecutable.getDispatchEntryOp().sym_name())});
        auto newSymbolRefAttr = builder.getSymbolRefAttr(
            comparisonExecutable.getName(),
            {builder.getSymbolRefAttr(
                comparisonExecutable.getDispatchEntryOp().sym_name())});
        entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
        duplicateExecutableOps.push_back(possiblyDuplicateExecutable);
        break;
      }
    }

    replaceEntryPointUses(moduleOp, entryPointRefReplacements);
    for (auto executableOp : duplicateExecutableOps) {
      executableOp.erase();
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
