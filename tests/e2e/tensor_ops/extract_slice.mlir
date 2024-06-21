func.func @extract_slice_strided() {
  %0 = tensor.empty() : tensor<500x750xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%0 : tensor<500x750xi32>) {
      ^bb0(%arg0 : i32):
        %2 = linalg.index 0 : index
        %3 = linalg.index 1 : index
        %4 = arith.index_cast %2 : index to i32
        %c750_i32 = arith.constant 750 : i32
        %5 = arith.muli %4, %c750_i32 : i32
        %6 = arith.index_cast %3 : index to i32
        %7 = arith.addi %5, %6 : i32
        linalg.yield %7 : i32
      } -> tensor<500x750xi32>
  %2 = tensor.extract_slice %1[20, 30] [50, 75] [2, 3]
      : tensor<500x750xi32> to tensor<50x75xi32>
  %3 = tensor.empty() : tensor<50x75xi32>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%3 : tensor<50x75xi32>) {
      ^bb0(%arg0 : i32) :
        %5 = linalg.index 0 : index
        %6 = linalg.index 1 : index
        %c20_i32 = arith.constant 20 : i32
        %c30_i32 = arith.constant 30 : i32
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %7 = arith.index_cast %6 : index to i32
        %8 = arith.muli %7, %c3_i32 : i32
        %9 = arith.addi %c30_i32, %8 : i32
        %10 = arith.index_cast %5 : index to i32
        %11 = arith.muli %10, %c2_i32 : i32
        %12 = arith.addi %c20_i32, %11 : i32
        %c750_i32 = arith.constant 750 : i32
        %13 = arith.muli %12, %c750_i32 : i32
        %14 = arith.addi %13, %9 : i32
        linalg.yield %14 : i32
     } -> tensor<50x75xi32>
  check.expect_eq(%2, %4) : tensor<50x75xi32>
  return
}

func.func public @extract_slice_i32_offset1_size2_stride1() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = util.optimization_barrier %1 : tensor<4xi32>
  // source [offsets] [sizes] [strides]
  %extracted_slice = tensor.extract_slice %2[1] [2] [1] : tensor<4xi32> to tensor<2xi32>
  check.expect_eq_const(%extracted_slice, dense<[2, 3]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func public @extract_slice_i64_offset1_size2_stride1() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %2 = util.optimization_barrier %1 : tensor<4xi64>
  // source [offsets] [sizes] [strides]
  %extracted_slice = tensor.extract_slice %2[1] [2] [1] : tensor<4xi64> to tensor<2xi64>
  check.expect_eq_const(%extracted_slice, dense<[2, 3]> : tensor<2xi64>) : tensor<2xi64>
  return
}

func.func public @extract_slice_i32_offset1_size2_stride2() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = util.optimization_barrier %1 : tensor<4xi32>
  // source [offsets] [sizes] [strides]
  %extracted_slice = tensor.extract_slice %2[1] [2] [2] : tensor<4xi32> to tensor<2xi32>
  check.expect_eq_const(%extracted_slice, dense<[2, 4]> : tensor<2xi32>) : tensor<2xi32>
  return
}
