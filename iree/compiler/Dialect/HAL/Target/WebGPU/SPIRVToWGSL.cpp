// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/SPIRVToWGSL.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
// #include "spirv-tools/libspirv.hpp"
#include "tint/tint.h"

#define DEBUG_TYPE "spirv-to-wgsl"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

llvm::Optional<std::string> compileSPIRVToWGSL(
    llvm::ArrayRef<uint32_t> spvBinary) {
  // LLVM_DEBUG(llvm::dbgs() << "Compiling SPIR-V to WGSL...\n");
  // llvm::dbgs() << "Compiling SPIR-V to WGSL...\n";

  // TODO(scotttodd): reroute to MLIR diagnostics?
  auto diag_printer = tint::diag::Printer::create(stderr, true);
  tint::diag::Formatter diag_formatter;

  // TODO(scotttodd): remove this copy (API for std::span or [uint8_t*, size]?)
  std::vector<uint32_t> binaryVector(spvBinary.size());
  std::memcpy(binaryVector.data(), spvBinary.data(),
              spvBinary.size() * sizeof(uint32_t));

  auto program =
      std::make_unique<tint::Program>(tint::reader::spirv::Parse(binaryVector));
  if (!program) {
    llvm::errs() << "Tint failed to parse SPIR-V program\n";
    return llvm::None;
  }

  if (program->Diagnostics().contains_errors()) {
    llvm::errs() << "Tint reported " << program->Diagnostics().error_count()
                 << " error(s) for a SPIR-V program, see diagnostics:\n";
    diag_formatter.format(program->Diagnostics(), diag_printer.get());
    return llvm::None;
  }

  if (!program->IsValid()) {
    llvm::errs() << "Tint parsed an invalid SPIR-V program\n";
    return llvm::None;
  }

  // return llvm::None;
  return "TODO";
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
