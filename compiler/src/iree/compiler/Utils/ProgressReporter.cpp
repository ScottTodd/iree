// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ProgressReporter.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

// static
llvm::sys::Mutex ProgressReporter::sMutex;

void ProgressReporter::runBeforePass(Pass *pass, Operation *op) {
  os << "\r: Running pass: '" << pass->getName() << "'";
  os.flush();

  // os << getThreadIndex() << ": Before '" << pass->getName() << "'\n";
}
void ProgressReporter::runAfterPass(Pass *pass, Operation *op) {
  // TODO(scotttodd): clear line
}
void ProgressReporter::runAfterPassFailed(Pass *pass, Operation *op) {
  // TODO(scotttodd): clear line
}

int ProgressReporter::getThreadIndex() {
  // Store a map of <thread id, index> and insert a new index into the map
  // whenever a new thread is encountered.
  //
  // It would be nice to match thread names set by `lib/Support/ThreadPool.cpp`:
  //   `llvm::get_thread_name(&name);`
  // See https://reviews.llvm.org/D144297, https://reviews.llvm.org/D147361
  // Name: `llvm-worker-{0}`
  //   1. extract the characters after the last `-`
  //   2. string to int

  llvm::thread::id id = llvm::this_thread::get_id();

  llvm::sys::ScopedLock lock(sMutex);

  if (threadIndices.contains(id)) {
    return threadIndices.at(id);
  }

  int index = threadIndices.size();
  threadIndices.insert({id, index});
  return index;
}

} // namespace iree_compiler
} // namespace mlir
