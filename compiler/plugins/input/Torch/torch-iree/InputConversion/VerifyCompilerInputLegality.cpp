// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-iree/InputConversion/PassDetail.h"
#include "torch-iree/InputConversion/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

namespace mlir::iree_compiler::TorchInput {

struct VerifyCompilerTorchInputLegalityPass
    : public VerifyCompilerTorchInputLegalityBase<
          VerifyCompilerTorchInputLegalityPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    conversionTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });
    conversionTarget.addIllegalDialect<torch::Torch::TorchDialect>();
    conversionTarget
        .addIllegalDialect<torch::TorchConversion::TorchConversionDialect>();
    conversionTarget
        .addIllegalDialect<mlir::torch::TMTensor::TMTensorDialect>();

    // We could call applyPartialConversion directly, but this helper creates
    // nice summary messages counting how many of each illegal op remains.
    if (failed(iree_compiler::verifyAllOperationsAreLegal(getOperation(),
                                                          conversionTarget))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyCompilerTorchInputLegality() {
  return std::make_unique<VerifyCompilerTorchInputLegalityPass>();
}

} // namespace mlir::iree_compiler::TorchInput
