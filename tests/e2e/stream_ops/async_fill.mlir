func.func public @async_fill_i64() {
//   %cst_1 = arith.constant dense<1> : tensor<1xi64>
//   %cst_2 = arith.constant dense<2> : tensor<1x1xi64>
//   %1 = util.optimization_barrier %cst_1 : tensor<1xi64>
//   %2 = util.optimization_barrier %cst_2 : tensor<1x1xi64>
//   %expanded_1 = tensor.expand_shape %1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>

  // check.expect_eq_const(%concat, dense<[[1,2,3,4]]> : tensor<1x4xi64>) : tensor<1x4xi64>
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


// util.func private @_expand_to_concat_i64() {
//   %c1 = arith.constant 1 : index
//   %c16 = arith.constant 16 : index
//   %c8 = arith.constant 8 : index
//   %c0 = arith.constant 0 : index
//   %c64 = arith.constant 64 : index
//   %c128 = arith.constant 128 : index
//   %c1_i32 = arith.constant 1 : i32
//   %c2_i32 = arith.constant 2 : i32
//   %__constant_tensor_2xi64__timepoint = util.global.load @__constant_tensor_2xi64__timepoint : !stream.timepoint
//   %__constant_tensor_2xi64 = util.global.load @__constant_tensor_2xi64 : !stream.resource<constant>
//   %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c8} => !stream.timepoint
//   %0 = stream.cmd.execute await(%result_timepoint) => with(%result as %arg0: !stream.resource<transient>{%c8}) {
//     stream.cmd.dispatch @__builtin_splat_i64::@embedded_elf_x86_64::@__builtin_splat_i64[%c1](%c1_i32 : i32) {
//       wo %arg0[%c0 for %c8] : !stream.resource<transient>{%c8}
//     }
//   } => !stream.timepoint
//   %1 = stream.timepoint.await %0 => %result : !stream.resource<transient>{%c8}
//   %2 = util.optimization_barrier %1 : !stream.resource<transient>
//   %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized : !stream.resource<transient>{%c8} => !stream.timepoint
//   %3 = stream.cmd.execute await(%result_timepoint_1) => with(%result_0 as %arg0: !stream.resource<transient>{%c8}) {
//     stream.cmd.dispatch @__builtin_splat_i64::@embedded_elf_x86_64::@__builtin_splat_i64[%c1](%c2_i32 : i32) {
//       wo %arg0[%c0 for %c8] : !stream.resource<transient>{%c8}
//     }
//   } => !stream.timepoint
//   %4 = stream.timepoint.await %3 => %result_0 : !stream.resource<transient>{%c8}
//   %5 = util.optimization_barrier %4 : !stream.resource<transient>
//   %6 = stream.resource.size %2 : !stream.resource<transient>
//   %7 = stream.resource.size %5 : !stream.resource<transient>
//   %result_2, %result_timepoint_3 = stream.resource.alloca uninitialized await(%__constant_tensor_2xi64__timepoint) => !stream.resource<external>{%c128} => !stream.timepoint
//   %8 = stream.cmd.execute await(%result_timepoint_3) => with(%2 as %arg0: !stream.resource<transient>{%6}, %5 as %arg1: !stream.resource<transient>{%7}, %__constant_tensor_2xi64 as %arg2: !stream.resource<constant>{%c64}, %result_2 as %arg3: !stream.resource<external>{%c128}) {
//     stream.cmd.copy %arg0[%c0], %arg3[%c0], %6 : !stream.resource<transient>{%6} -> !stream.resource<external>{%c128}
//     stream.cmd.concurrent {
//       stream.cmd.copy %arg1[%c0], %arg3[%c8], %7 : !stream.resource<transient>{%7} -> !stream.resource<external>{%c128}
//       stream.cmd.copy %arg2[%c0], %arg3[%c64], %c16 : !stream.resource<constant>{%c64} -> !stream.resource<external>{%c128}
//     }
//   } => !stream.timepoint
//   %9 = stream.timepoint.await %8 => %result_2 : !stream.resource<external>{%c128}
//   %10 = stream.resource.subview %9[%c64] : !stream.resource<external>{%c128} -> !stream.resource<external>{%c16}
//   %11 = stream.resource.subview %9[%c0] : !stream.resource<external>{%c128} -> !stream.resource<external>{%c16}
//   %12 = stream.tensor.export %10 : tensor<2xi64> in !stream.resource<external>{%c16} -> tensor<2xi64>
//   %13 = stream.tensor.export %11 : tensor<2xi64> in !stream.resource<external>{%c16} -> tensor<2xi64>
//   check.expect_eq(%13, %12) : tensor<2xi64>
//   util.return
// }
