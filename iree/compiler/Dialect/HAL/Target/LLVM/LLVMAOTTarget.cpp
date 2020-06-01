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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/dylib_executable_def_generated.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetSelect.h"

// TODO(scotttodd): replace with actual LLVM AOT compilation
#include "iree/compiler/Dialect/HAL/Target/LLVM/TestLibrary.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMAOTTargetBackend final : public LLVMBaseTargetBackend {
 public:
  LLVMAOTTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options, such as by arch/etc.
  std::string name() const override { return "llvm-aot*"; }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder& executableBuilder) override {
    // TODO(scotttodd): check if this is needed here, or just in base
    // LLVM is not thread safe and currently translation shares an LLVMContext.
    // Since we serialize executables from multiple threads we have to take a
    // global lock here.
    static llvm::sys::SmartMutex<true> mutex;
    llvm::sys::SmartScopedLock<true> lock(mutex);

    iree::DyLibExecutableDefT dyLibExecutableDef;

    auto llvmModule = translateTargetOp(targetOp, [&](std::string funcName) {
      dyLibExecutableDef.entry_points.push_back(funcName);
    });

    if (!llvmModule) {
      return failure();
    }

    // Proof of concept test:
    //   * [done] cc_embed_data of test.dll
    //   * [done] add bytes to flatbuffer
    //   * [done] extract bytes from flatbuffer in dylib_executable
    //   * [done] write bytes to tmp file
    //   * [done] load tmp file absolute path via DynamicLibrary

    // TODO(scotttodd): replace with actual LLVM AOT compilation
    const auto* testFileToc = TestLibrary_create();
    dyLibExecutableDef.library_embedded.assign(
        testFileToc->data, testFileToc->data + testFileToc->size);

    // // Serialize LLVM module.
    // std::string bufferString;
    // llvm::raw_string_ostream ostream(bufferString);
    // llvmModule->print(ostream, nullptr);
    // ostream.flush();

    // Creates executable bytes.
    // llvmIrExecutableDef.llvmir_module = {bufferString.begin(),
    //                                      bufferString.end()};

    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::DyLibExecutableDef::Pack(fbb, &dyLibExecutableDef);
    iree::FinishDyLibExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::DyLib),
        std::move(bytes));

    return success();
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("llvm-aot", [=]() {
    // Initalize registered targets.
    llvm::InitializeNativeTarget();
    // llvm::InitializeNativeTargetAsmPrinter();
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
