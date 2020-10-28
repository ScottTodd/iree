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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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
