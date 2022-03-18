// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but without dialects we don't care about.

#ifndef IREE_TOOLS_INIT_MLIR_DIALECTS_H_
#define IREE_TOOLS_INIT_MLIR_DIALECTS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerMlirDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<AffineDialect,
                  arm_neon::ArmNeonDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  emitc::EmitCDialect,
                  func::FuncDialect,
                  gpu::GPUDialect,
                  linalg::LinalgDialect,
                  LLVM::LLVMDialect,
                  math::MathDialect,
                  memref::MemRefDialect,
                  mlir::arith::ArithmeticDialect,
                  quant::QuantizationDialect,
                  scf::SCFDialect,
                  shape::ShapeDialect,
                  spirv::SPIRVDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect,
                  vector::VectorDialect>();
  // clang-format on
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
}

}  // namespace mlir

#endif  // IREE_TOOLS_INIT_MLIR_DIALECTS_H_
