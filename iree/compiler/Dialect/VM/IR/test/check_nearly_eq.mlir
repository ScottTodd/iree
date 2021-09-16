// NORUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s

// CHECK-LABEL: @check_folds
vm.module @check_folds {
  // CHECK-LABEL: @check_nearly_eq_f32
  vm.func @check_nearly_eq_f32(%arg0 : f32, %arg1 : f32) {
    vm.check.nearly_eq %arg0, %arg1, "expected nearly eq" : f32
    vm.return
  }
}
