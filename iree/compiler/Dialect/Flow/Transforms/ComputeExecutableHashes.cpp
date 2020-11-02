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

#include <iostream>  // DO NOT SUBMIT

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

class ExecutableFuncOpHashAnalysis {
 public:
  explicit ExecutableFuncOpHashAnalysis(Operation *op) {
    ExecutableOp executableOp = cast<ExecutableOp>(op);

    auto module = executableOp.getInnerModule();
    auto funcs = llvm::to_vector<1>(module.getOps<FuncOp>());
    auto func = *funcs.begin();

    std::string funcStr;
    llvm::raw_string_ostream sstream(funcStr);
    auto funcRegion = func.getCallableRegion();
    for (auto &block : funcRegion->getBlocks()) {
      block.print(sstream);
    }
    sstream.flush();
    hash = llvm::hash_value(funcStr);

    // --------------------
    // DEBUG, DO NOT SUBMIT
    auto parentModuleOp = dyn_cast<ModuleOp>(executableOp.getParentOp());
    auto siblingExecutableOps =
        llvm::to_vector<8>(parentModuleOp.getOps<ExecutableOp>());
    for (int i = 0; i < siblingExecutableOps.size(); ++i) {
      auto siblingExecutableOp = siblingExecutableOps[i];
      if (executableOp == siblingExecutableOp) {
        std::cerr << "Computed hash for executable index " << i << std::endl;
        break;
      }
    }
    // DEBUG, DO NOT SUBMIT
    // --------------------
  }

  llvm::hash_code hash;
};

}  // namespace

class ComputeExecutableHashesPass
    : public PassWrapper<ComputeExecutableHashesPass,
                         OperationPass<ExecutableOp>> {
 public:
  ComputeExecutableHashesPass() = default;

  void runOnOperation() override {
    auto &funcOpHashAnalysis = getAnalysis<ExecutableFuncOpHashAnalysis>();
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<OperationPass<ExecutableOp>>
createComputeExecutableHashesPass() {
  return std::make_unique<ComputeExecutableHashesPass>();
}

static PassRegistration<ComputeExecutableHashesPass> pass(
    "iree-flow-compute-executable-hashes",
    "Computes and caches hashes of executable ops");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
