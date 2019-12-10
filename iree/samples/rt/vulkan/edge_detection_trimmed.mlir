// Copyright 2019 Google LLC
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

// Image edge detection module generated by iree/colab/edge_detection.ipynb,
// then trimmed manually to use simpler semantics.
//
// Input : a single 128x128 pixel image as a tensor<1x128x128x1xf32>, with pixels in [0.0, 1.0]
// Output: a single image in the same format after running edge detection

func @edge_detect_sobel_operator(%arg0: tensor<1x128x128x1xf32>) -> tensor<1x128x128x1xf32>
    attributes { iree.module.export } {
  %0 = xla_hlo.constant dense<[[[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]], [[[-2.000000e+00]], [[0.000000e+00]], [[2.000000e+00]]], [[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>
  %1 = xla_hlo.constant dense<[[[[1.000000e+00]], [[2.000000e+00]], [[1.000000e+00]]], [[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]], [[[-1.000000e+00]], [[-2.000000e+00]], [[-1.000000e+00]]]]> : tensor<3x3x1x1xf32>
  %2 = "xla_hlo.conv"(%arg0, %0) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x128x128x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x128x128x1xf32>
  %3 = xla_hlo.mul %2, %2 : tensor<1x128x128x1xf32>
  %4 = "xla_hlo.conv"(%arg0, %1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x128x128x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x128x128x1xf32>
  %5 = xla_hlo.mul %4, %4 : tensor<1x128x128x1xf32>
  %6 = xla_hlo.add %3, %5 : tensor<1x128x128x1xf32>
  %7 = "xla_hlo.sqrt"(%6) : (tensor<1x128x128x1xf32>) -> tensor<1x128x128x1xf32>
  return %7 : tensor<1x128x128x1xf32>
}
