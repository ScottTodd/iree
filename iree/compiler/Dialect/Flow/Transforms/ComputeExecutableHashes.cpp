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
    auto executableOp = dyn_cast<ExecutableOp>(op);
    auto module = executableOp.getInnerModule();

    // Assume only one FuncOp on the module.
    auto funcs = llvm::to_vector<1>(module.getOps<FuncOp>());
    auto func = *funcs.begin();

    // Print the blocks of the function into a string and hash it.
    //
    // We'd prefer to have a more native (and efficient) way of comparing two
    // ops, but this works well enough for our current uses.
    //
    // This includes the full function signature (arguments and their types,
    // output(s) and their types) and all nested ops with their attributes and
    // other printed properties.
    std::string funcStr;
    llvm::raw_string_ostream sstream(funcStr);
    auto funcRegion = func.getCallableRegion();
    for (auto &block : funcRegion->getBlocks()) {
      block.print(sstream);
    }
    sstream.flush();

    std::cerr << funcStr << std::endl;

    llvm::hash_code hash = llvm::hash_value(funcStr);
    // TODO(scotttodd): solution that doesn't require editing IR?
    Builder builder(module.getContext());
    executableOp.setAttr("func_hash", builder.getI64IntegerAttr(hash));
  }
};

}  // namespace

class ComputeExecutableHashesPass
    : public PassWrapper<ComputeExecutableHashesPass,
                         OperationPass<ExecutableOp>> {
 public:
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
