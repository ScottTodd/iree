func.func public @async_fill_i64() {
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
  return
}

// // CHECK-LABEL: @FlattenFullFillToSplatUnsafe
// util.func private @FlattenFullFillToSplatUnsafe(%arg0: index, %arg1: i32, %arg2: !hal.buffer_view) -> !stream.resource<*> {
//   %c0 = arith.constant 0 : index
//   // CHECK: stream.tensor.import
//   %target = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<8xi32> in !stream.resource<*>{%arg0}
//   // CHECK: stream.async.fill
//   %0 = stream.async.fill %arg1, %target[%c0 to %arg0 for %arg0] : i32 -> %target as !stream.resource<*>{%arg0}
//   util.return %0 : !stream.resource<*>
// }


// util.func public @builtinFillI64(%res: !stream.resource<*>, %size: index, %value: i64, %byte_offset: index, %byte_end: index, %byte_length: index) -> !stream.resource<*> {
//   // CHECK: %[[COUNT:.+]] = arith.divui %[[BYTE_LENGTH]], %c8
//   // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_fill_i64::@__builtin_fill_i64[%[[COUNT]]](%[[RES]][%[[BYTE_OFFSET]] to %[[BYTE_END]] for %[[BYTE_LENGTH]]], %[[VALUE]], %[[COUNT]]) : (!stream.resource<*>{%[[SIZE]]}, i64, index) -> %[[RES]]{%[[SIZE]]}
//   %0 = stream.async.fill %value, %res[%byte_offset to %byte_end for %byte_length] : i64 -> %arg0 as !stream.resource<*>{%size}
//   // CHECK: util.return %[[RET]]
//   util.return %0 : !stream.resource<*>
// }
