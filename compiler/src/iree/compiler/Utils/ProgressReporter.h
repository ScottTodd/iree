// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PROGRESSREPORTER_H_
#define IREE_COMPILER_UTILS_PROGRESSREPORTER_H_

#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/thread.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
namespace iree_compiler {

// Reports compilation progress to an output stream.
//
// Usage:
//   pm.addInstrumentation(std::make_unique<ProgressReporter>(llvm::dbgs()));
class ProgressReporter : public PassInstrumentation {
public:
  ProgressReporter(llvm::raw_ostream &os) : os(os) {}
  ~ProgressReporter() override = default;

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

private:
  // Gets a positive index for the current thread.
  // Thread index assignments are arbitrary.
  int getThreadIndex();

  llvm::raw_ostream &os;

  static llvm::sys::Mutex sMutex; // Lock for threadIndices.
  llvm::DenseMap<llvm::thread::id, int> threadIndices;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_PROGRESSREPORTER_H_
