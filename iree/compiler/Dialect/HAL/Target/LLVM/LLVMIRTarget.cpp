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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRTarget.h"

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMIRTargetBackend final : public LLVMBaseTargetBackend {
 public:
  LLVMIRTargetBackend(LLVMTargetOptions options)
      : LLVMBaseTargetBackend(options) {}

  // NOTE: we could vary this based on the options, such as by arch/etc.
  std::string name() const override { return "llvm-ir*"; }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder& executableBuilder) override {
    // TODO(scotttodd): check if this is needed here, or just in base
    // LLVM is not thread safe and currently translation shares an LLVMContext.
    // Since we serialize executables from multiple threads we have to take a
    // global lock here.
    static llvm::sys::SmartMutex<true> mutex;
    llvm::sys::SmartScopedLock<true> lock(mutex);

    iree::LLVMIRExecutableDefT llvmIrExecutableDef;

    auto llvmModule = translateTargetOp(targetOp, [&](std::string funcName) {
      llvmIrExecutableDef.entry_points.push_back(funcName);
    });

    if (!llvmModule) {
      return failure();
    }

    // Serialize LLVM module.
    std::string bufferString;
    llvm::raw_string_ostream ostream(bufferString);
    llvmModule->print(ostream, nullptr);
    ostream.flush();

    // Creates executable bytes.
    llvmIrExecutableDef.llvmir_module = {bufferString.begin(),
                                         bufferString.end()};

    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::LLVMIRExecutableDef::Pack(fbb, &llvmIrExecutableDef);
    iree::FinishLLVMIRExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
        std::move(bytes));

    return success();
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMIRTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("llvm-ir", [=]() {
    // Initalize registered targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return std::make_unique<LLVMIRTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
