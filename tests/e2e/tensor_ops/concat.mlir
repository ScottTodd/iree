// func.func public @concat_i64() {
//   %cst_1 = arith.constant dense<1> : tensor<1xi64>
//   %cst_2 = arith.constant dense<2> : tensor<1x1xi64>
//   %cst_3 = arith.constant dense<3> : tensor<1x1xi64>
//   %cst_4 = arith.constant dense<4> : tensor<1x1xi64>
//   %1 = util.optimization_barrier %cst_1 : tensor<1xi64>
//   %2 = util.optimization_barrier %cst_2 : tensor<1x1xi64>
//   %3 = util.optimization_barrier %cst_3 : tensor<1x1xi64>
//   %4 = util.optimization_barrier %cst_4 : tensor<1x1xi64>
//   //
//   %expanded_1 = tensor.expand_shape %1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>

//   // DO NOT SUBMIT
//   // This exercises __builtin_splat_i64
//   // Need a different test to exercise __builtin_fill_i64
//   // Should also add tests for
//   //   * other data types
//   //   * builtins directly (starting from stream)
//   %concat = tensor.concat dim(1) %expanded_1, %2, %3, %4 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
//   check.expect_eq_const(%concat, dense<[[1,2,3,4]]> : tensor<1x4xi64>) : tensor<1x4xi64>
//   return
// }


// func.func public @expand_to_concat_i64() {
//   %cst_1 = arith.constant dense<1> : tensor<1xi64>
//   %cst_2 = arith.constant dense<2> : tensor<1x1xi64>
//   %cst_3 = arith.constant dense<3> : tensor<1x1xi64>
//   %cst_4 = arith.constant dense<4> : tensor<1x1xi64>
//   %1 = util.optimization_barrier %cst_1 : tensor<1xi64>
//   %2 = util.optimization_barrier %cst_2 : tensor<1x1xi64>
//   %3 = util.optimization_barrier %cst_3 : tensor<1x1xi64>
//   %4 = util.optimization_barrier %cst_4 : tensor<1x1xi64>
//   %expanded_1 = tensor.expand_shape %1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
//   // This exercises __builtin_splat_i64
//   %concat = tensor.concat dim(1) %expanded_1, %2, %3, %4 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
//   check.expect_eq_const(%concat, dense<[[1,2,3,4]]> : tensor<1x4xi64>) : tensor<1x4xi64>
//   return
// }


func.func public @expand_to_concat_i64() {
  %cst_1 = arith.constant dense<1> : tensor<1xi64>
  %cst_2 = arith.constant dense<2> : tensor<1xi64>
  %1 = util.optimization_barrier %cst_1 : tensor<1xi64>
  %2 = util.optimization_barrier %cst_2 : tensor<1xi64>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  check.expect_eq_const(%concat, dense<[1,2]> : tensor<2xi64>) : tensor<2xi64>
  return
}
