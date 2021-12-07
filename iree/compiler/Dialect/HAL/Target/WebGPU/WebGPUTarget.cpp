// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/WebGPUTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

WebGPUTargetOptions getWebGPUTargetOptionsFromFlags() {
  static llvm::cl::opt<bool> clDebugSymbols(
      "iree-webgpu-debug-symbols",
      llvm::cl::desc(
          "Include debug information like variable names in outputs"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> clWebGPUKeepShaderModules(
      "iree-webgpu-keep-shader-modules",
      llvm::cl::desc("Save shader modules to disk separately"),
      llvm::cl::init(false));

  WebGPUTargetOptions targetOptions;
  targetOptions.keepShaderModules = clWebGPUKeepShaderModules;

  return targetOptions;
}

// TODO(scotttodd): provide a proper target environment for WebGPU.
static spirv::TargetEnvAttr getWebGPUTargetEnv(MLIRContext *context) {
  // TODO(scotttodd): find list of SPIR-V extensions supported by WebGPU/WGSL
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
}

class WebGPUTargetBackend : public TargetBackend {
 public:
  WebGPUTargetBackend(WebGPUTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'webgpu-v2'.
  std::string name() const override { return "webgpu"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getIdentifier("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildSPIRVCodegenPassPipeline(passManager);
    // TODO(scotttodd): additional passes for WebGPU/WGSL
    //                  (here or during serialization?)
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spirvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
    if (!llvm::hasSingleElement(spirvModuleOps)) {
      return variantOp.emitError()
             << "should only contain exactly one spv.module op";
    }
    auto spvModuleOp = *spirvModuleOps.begin();

    // Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary)) || spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spv.module";
    }
    if (options_.keepShaderModules) {
      saveSpirvBinary(variantOp, spvBinary);
    }

    // TODO(scotttodd): Cross compile SPIR-V to WGSL source code.

    // TODO(scotttodd): Pack the WGSL and metadata into a flatbuffer.

    // TODO(scotttodd): Add the binary data to the target executable.

    return variantOp.emitError("WebGPU/WGSL serialization not yet implemented");
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(
        getExecutableTarget(context, getWebGPUTargetEnv(context)));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context, spirv::TargetEnvAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getIdentifier(spirv::getTargetEnvAttrName()),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("webgpu"), b.getStringAttr("webgpu-wgsl-fb"),
        configAttr);
  }

  void saveSpirvBinary(IREE::HAL::ExecutableVariantOp variantOp,
                       ArrayRef<uint32_t> binary) {
    llvm::SmallString<32> filePath;
    if (std::error_code error = llvm::sys::fs::createTemporaryFile(
            variantOp.getName(), "spv", filePath)) {
      llvm::errs() << "failed to generate file for SPIR-V binary: "
                   << error.message();
      return;
    }
    std::error_code error;
    auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                       llvm::sys::fs::OF_None);
    if (error) {
      llvm::errs() << "failed to open file for SPIR-V binary '" << filePath
                   << "': " << error.message();
      return;
    }

    mlir::emitRemark(variantOp.getLoc())
        << "SPIR-V binary for " << variantOp.getName() << " preserved:\n"
        << "    " << filePath;
    file->os().write(reinterpret_cast<const char *>(binary.data()),
                     binary.size() * sizeof(uint32_t));
    file->keep();
  }

  WebGPUTargetOptions options_;
};

void registerWebGPUTargetBackends(
    std::function<WebGPUTargetOptions()> queryOptions) {
  getWebGPUTargetOptionsFromFlags();
  auto backendFactory = [=]() {
    return std::make_shared<WebGPUTargetBackend>(queryOptions());
  };
  // #hal.device.target<"webgpu", ...
  static TargetBackendRegistration registration0("webgpu", backendFactory);
  // #hal.executable.target<"webgpu-wgsl", ...
  static TargetBackendRegistration registration1("webgpu-wgsl", backendFactory);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
