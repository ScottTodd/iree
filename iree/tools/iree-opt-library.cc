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

// Re-entrant functions for iree-opt, forked from iree-opt-main.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "iree/tools/init_dialects.h"
#include "iree/tools/init_passes.h"
#include "iree/tools/init_targets.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"

static bool initialized = false;
static mlir::DialectRegistry registry;

extern "C" {

void initialize() {
  if (initialized) {
    return;
  }

  // TODO(scotttodd): InitLLVM and keep the object alive?

  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  initialized = true;
}

int run(int argc, char **argv) {
  // TODO(scotttodd): see if this is an issue (has static state), can maybe omit
  llvm::InitLLVM y(argc, argv);

  if (failed(MlirOptMain(argc, argv, "IREE modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}

int main(int arc, char **argv) {
  initialize();
  return 0;
}

}  // extern "C"
