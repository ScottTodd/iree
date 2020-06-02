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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// static
void LLVMBaseTargetBackend::createInvocationFunc(const std::string& name,
                                                 llvm::Module* module) {
  // TODO(ataei): This is written as a stub in LLVM IR. It would be easier to
  // have this using MLIR and lower it to LLVM like the dispatch function
  // implementation is.

  auto& ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto var_func = module->getFunction(name);

  auto new_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  auto new_name = "invoke_" + name;
  auto func_cst = module->getOrInsertFunction(new_name, new_type);
  llvm::Function* interface_func =
      llvm::cast<llvm::Function>(func_cst.getCallee());

  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interface_func);
  builder.SetInsertPoint(bb);
  llvm::Value* argList = interface_func->arg_begin();
  llvm::SmallVector<llvm::Value*, 8> args;
  args.reserve(llvm::size(var_func->args()));
  for (auto& indexedArg : llvm::enumerate(var_func->args())) {
    llvm::Value* arg_index = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
    llvm::Value* arg_ptr_ptr = builder.CreateGEP(argList, arg_index);
    llvm::Value* arg_ptr = builder.CreateLoad(arg_ptr_ptr);
    arg_ptr = builder.CreateBitCast(
        arg_ptr, indexedArg.value().getType()->getPointerTo());
    llvm::Value* arg = builder.CreateLoad(arg_ptr);
    args.push_back(arg);
  }
  builder.CreateCall(var_func, args);
  builder.CreateRetVoid();
}

LLVMBaseTargetBackend::LLVMBaseTargetBackend(LLVMTargetOptions options)
    : options_(std::move(options)) {}

void LLVMBaseTargetBackend::buildTranslationPassPipeline(
    ExecutableTargetOp targetOp, OpPassManager& passManager) {
  buildLLVMTransformPassPipeline(passManager);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
