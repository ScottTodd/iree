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
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_BASE_TARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_BASE_TARGET_H_

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMBaseTargetBackend : public TargetBackend {
 public:
  LLVMBaseTargetBackend(LLVMTargetOptions options);

  // Adds a sequence of passes to a given pass manager that progressively lower
  // from HLO to LLVM through the linalg dialect.
  void buildTranslationPassPipeline(ExecutableTargetOp targetOp,
                                    OpPassManager& passManager) override;

 protected:
  // Translates the given |targetOp| executable to an llvm::Module.
  // Emits an error on |targetOp| and returns nullptr if translation fails.
  // Invokes |addEntryPointFunction| for each entry point function name
  // encountered during translation.
  std::unique_ptr<llvm::Module> translateTargetOp(
      IREE::HAL::ExecutableTargetOp targetOp,
      std::function<void(std::string)> addEntryPointFunction);

 private:
  LLVMTargetOptions options_;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_BASE_TARGET_H_
