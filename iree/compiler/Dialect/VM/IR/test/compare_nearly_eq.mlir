// NORUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s

// CHECK-LABEL: @cmp_eq_f32_near_folds
vm.module @cmp_eq_f32_near_folds {
  // CHECK-LABEL: @eq_near_f32
  vm.func @eq_near_f32(%arg0 : f32, %arg1 : f32) -> i32 {
    %cmp = vm.cmp.eq.f32.near %arg0, %arg1 : f32
    vm.return %cmp : i32
  }
}
