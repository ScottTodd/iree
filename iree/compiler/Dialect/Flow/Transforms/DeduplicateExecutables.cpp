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
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

// TODO(scotttodd): pass statistics (number of executables deduped)
// https://mlir.llvm.org/docs/PassManagement/#pass-statistics

#include <iostream>  // DO NOT SUBMIT

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

bool areExecutablesEquivalent(ExecutableOp lhs, ExecutableOp rhs) {
  auto lhsModule = lhs.getInnerModule();
  auto rhsModule = rhs.getInnerModule();
  auto lhsFuncs = llvm::to_vector<1>(lhsModule.getOps<FuncOp>());
  auto rhsFuncs = llvm::to_vector<1>(rhsModule.getOps<FuncOp>());
  auto lhsFunc = *lhsFuncs.begin();
  auto rhsFunc = *rhsFuncs.begin();

  // std::cerr << "lhs func:" << std::endl;
  // lhsFunc.dump();
  // std::cerr << "rhs func:" << std::endl;
  // rhsFunc.dump();

  std::string lhsStr;
  llvm::raw_string_ostream lhsSstream(lhsStr);
  auto lhsFuncRegion = lhsFunc.getCallableRegion();
  for (auto &block : lhsFuncRegion->getBlocks()) {
    block.print(lhsSstream);
  }
  lhsSstream.flush();
  llvm::hash_code lhsHash = llvm::hash_value(lhsStr);
  // std::cerr << "lhsFuncRegion: " << std::endl;
  // std::cerr << lhsStr;

  std::string rhsStr;
  llvm::raw_string_ostream rhsSstream(rhsStr);
  auto rhsFuncRegion = rhsFunc.getCallableRegion();
  for (auto &block : rhsFuncRegion->getBlocks()) {
    block.print(rhsSstream);
  }
  rhsSstream.flush();
  llvm::hash_code rhsHash = llvm::hash_value(rhsStr);
  // std::cerr << "rhsFuncRegion: " << std::endl;
  // std::cerr << rhsStr;

  // if (lhsStr == rhsStr) {
  //   // std::cerr << "equivalent!" << std::endl;
  //   return true;
  // }

  if (lhsHash == rhsHash) {
    return true;
  }

  return false;
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
    auto moduleOp = getOperation();
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());

    SmallVector<ExecutableOp, 3> duplicateExecutableOps;
    DenseMap<Attribute, Attribute> entryPointRefReplacements;

    for (auto executableOp : executableOps) {
      auto duplicateOpSym =
          executableOp.getAttrOfType<SymbolRefAttr>("duplicate");
      if (!duplicateOpSym) {
        continue;
      }

      auto duplicateOp =
          dyn_cast<ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
              moduleOp, duplicateOpSym.getLeafReference()));

      auto oldSymbolRefAttr = builder.getSymbolRefAttr(
          executableOp.getName(),
          {builder.getSymbolRefAttr(
              executableOp.getDispatchEntryOp().sym_name())});
      auto newSymbolRefAttr = builder.getSymbolRefAttr(
          duplicateOp.getName(),
          {builder.getSymbolRefAttr(
              duplicateOp.getDispatchEntryOp().sym_name())});
      entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      duplicateExecutableOps.push_back(executableOp);
    }

    // // For each executable, find the first executable which it is equivalent
    // to.
    // // Iteration order:
    // //   3 == 0 ? no
    // //   3 == 1 ? yes -> mark and move on to 2
    // //   2 == 0 ? no
    // //   2 == 1 ? yes -> mark and move on to 1
    // //   1 == 0 ? no -> not a duplicate, keep it
    // //   Done iterating. 0 and 1 stay, 2 and 3 are duplicates of 1.
    // for (int i = executableOps.size() - 1; i >= 0; --i) {
    //   auto possiblyDuplicateExecutable = executableOps[i];
    //   for (int j = 0; j < i; ++j) {
    //     // std::cerr << "i: " << i << ", j: " << j << std::endl;

    //     auto comparisonExecutable = executableOps[j];
    //     if (!areExecutablesEquivalent(possiblyDuplicateExecutable,
    //                                   comparisonExecutable)) {
    //       continue;
    //     }

    //     // std::cerr << "Duplicate! replacing " << i << " with " << j <<
    //     // std::endl;

    //     // Add to replacement table and break to move to the next possible
    //     dup. auto oldSymbolRefAttr = builder.getSymbolRefAttr(
    //         possiblyDuplicateExecutable.getName(),
    //         {builder.getSymbolRefAttr(
    //             possiblyDuplicateExecutable.getDispatchEntryOp().sym_name())});
    //     auto newSymbolRefAttr = builder.getSymbolRefAttr(
    //         comparisonExecutable.getName(),
    //         {builder.getSymbolRefAttr(
    //             comparisonExecutable.getDispatchEntryOp().sym_name())});
    //     entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
    //     duplicateExecutableOps.push_back(possiblyDuplicateExecutable);
    //     break;
    //   }
    // }

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
