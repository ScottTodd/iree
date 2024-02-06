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

// expected-error-re@+1 {{The following illegal operations still remain:{{.*}}torch.constant.int (count: 1){{.*}}torch.operator (count: 1)}}
module {
func.func @unconverted_torch_ops() {
  // expected-error@+1 {{'torch.operator' op : illegal op still exists}}
  torch.operator "test_unconverted_op" () : () -> ()
  // expected-error@+1 {{'torch.constant.int' op : illegal op still exists}}
  %int = torch.constant.int 1
  return
}
}
