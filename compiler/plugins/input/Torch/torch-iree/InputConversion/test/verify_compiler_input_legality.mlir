// RUN: iree-opt --split-input-file \
// RUN:   --torch-iree-verify-compiler-input-legality \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Dialects that IREE natively supports should just be passed through unchanged.
// CHECK-LABEL: func.func @abs(
module {
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  // CHECK: %[[RESULT:.+]] = math.absf %[[INPUT:.+]] : tensor<f32>
  %result = math.absf %input : tensor<f32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<f32>
  return %result : tensor<f32>
}
}

// -----

// expected-error@+1 {{one or more illegal operations were found in the compiler input}}
module {
func.func @unknown_onnx() {
  // expected-note@+1 {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  torch.operator "test_unconverted_op" () : () -> ()
  return
}
}
