// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/SPIRVToWGSL.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "spirv-tools/libspirv.hpp"  // for hacks
#include "tint/tint.h"

#define DEBUG_TYPE "spirv-to-wgsl"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

// This is the .spvasm from iree/samples/models/simple_abs.mlir, but with
// a duplicate `OpDecorate %_struct_2 Block` removed to hack around
// "error: can't handle a struct with more than one decoration: struct 2 has 2"
std::string kHardcodedShader =
    "; SPIR-V\n"
    "; Version: 1.0\n"
    "; Generator: Khronos; 22\n"
    "; Bound: 19\n"
    "; Schema: 0\n"
    "               OpCapability Shader\n"
    "               OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
    "         %17 = OpExtInstImport \"GLSL.std.450\"\n"
    "               OpMemoryModel Logical GLSL450\n"
    "               OpEntryPoint GLCompute %d0 \"d0\"\n"
    "               OpExecutionMode %d0 LocalSize 1 1 1\n"
    "               OpName %__resource_var_0_0_ \"__resource_var_0_0_\"\n"
    "               OpName %__resource_var_0_1_ \"__resource_var_0_1_\"\n"
    "               OpName %d0 \"d0\"\n"
    "               OpDecorate %_runtimearr_float ArrayStride 4\n"
    "               OpMemberDecorate %_struct_2 0 Offset 0\n"
    "               OpDecorate %_struct_2 Block\n"
    "               OpDecorate %__resource_var_0_0_ Binding 0\n"
    "               OpDecorate %__resource_var_0_0_ DescriptorSet 0\n"
    "               OpDecorate %__resource_var_0_1_ Binding 1\n"
    "               OpDecorate %__resource_var_0_1_ DescriptorSet 0\n"
    "      %float = OpTypeFloat 32\n"
    "%_runtimearr_float = OpTypeRuntimeArray %float\n"
    "  %_struct_2 = OpTypeStruct %_runtimearr_float\n"
    "%_ptr_StorageBuffer__struct_2 = OpTypePointer StorageBuffer %_struct_2\n"
    "%__resource_var_0_0_ = OpVariable %_ptr_StorageBuffer__struct_2 "
    "StorageBuffer\n"
    "%__resource_var_0_1_ = OpVariable %_ptr_StorageBuffer__struct_2 "
    "StorageBuffer\n"
    "       %void = OpTypeVoid\n"
    "          %7 = OpTypeFunction %void\n"
    "       %uint = OpTypeInt 32 0\n"
    "     %uint_0 = OpConstant %uint 0\n"
    "%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float\n"
    "         %d0 = OpFunction %void None %7\n"
    "         %10 = OpLabel\n"
    "         %14 = OpAccessChain %_ptr_StorageBuffer_float "
    "%__resource_var_0_0_ %uint_0 %uint_0\n"
    "         %15 = OpLoad %float %14\n"
    "         %16 = OpExtInst %float %17 FAbs %15\n"
    "         %18 = OpAccessChain %_ptr_StorageBuffer_float "
    "%__resource_var_0_1_ %uint_0 %uint_0\n"
    "               OpStore %18 %16\n"
    "               OpReturn\n"
    "               OpFunctionEnd\n";

}  // namespace

llvm::Optional<std::string> compileSPIRVToWGSL(
    llvm::ArrayRef<uint32_t> spvBinary) {
  // LLVM_DEBUG(llvm::dbgs() << "Compiling SPIR-V to WGSL...\n");
  // llvm::dbgs() << "Compiling SPIR-V to WGSL...\n";

  // TODO(scotttodd): reroute to MLIR diagnostics?
  auto diagPrinter = tint::diag::Printer::create(stderr, true);
  tint::diag::Formatter diagFormatter;

  // TODO(scotttodd): remove this copy (API for std::span or [uint8_t*, size]?)
  // std::vector<uint32_t> binaryVector(spvBinary.size());
  // std::memcpy(binaryVector.data(), spvBinary.data(),
  //             spvBinary.size() * sizeof(uint32_t));

  // HACK
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::vector<uint32_t> binaryVector;
  if (!tools.Assemble(kHardcodedShader.data(), &binaryVector,
                      SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)) {
    return llvm::None;
  }
  // HACK

  auto program =
      std::make_unique<tint::Program>(tint::reader::spirv::Parse(binaryVector));
  if (!program) {
    llvm::errs() << "Tint failed to parse SPIR-V program\n";
    return llvm::None;
  }

  if (program->Diagnostics().contains_errors()) {
    llvm::errs() << "Tint reported " << program->Diagnostics().error_count()
                 << " error(s) for a SPIR-V program, see diagnostics:\n";
    diagFormatter.format(program->Diagnostics(), diagPrinter.get());
    return llvm::None;
  }

  if (!program->IsValid()) {
    llvm::errs() << "Tint parsed an invalid SPIR-V program\n";
    return llvm::None;
  }

  // TODO(scotttodd): Refine this set of transforms
  tint::transform::Manager transformManager;
  tint::transform::DataMap transformInputs;
  transformInputs.Add<tint::transform::FirstIndexOffset::BindingPoint>(0, 0);
  transformManager.Add<tint::transform::FirstIndexOffset>();
  transformManager.Add<tint::transform::FoldTrivialSingleUseLets>();

  auto output = transformManager.Run(program.get(), std::move(transformInputs));
  if (!output.program.IsValid()) {
    llvm::errs() << "Tint transforms failed on the parsed SPIR-V program\n";
    diagFormatter.format(output.program.Diagnostics(), diagPrinter.get());
    return llvm::None;
  }

  tint::writer::wgsl::Options genOptions;
  auto result = tint::writer::wgsl::Generate(&output.program, genOptions);
  if (!result.success) {
    llvm::errs() << "Tint failed to generate WGSL: " << result.error << "\n";
    return llvm::None;
  }

  return result.wgsl;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
