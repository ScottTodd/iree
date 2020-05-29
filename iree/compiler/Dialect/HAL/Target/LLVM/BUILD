# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

cc_library(
    name = "LLVMBase",
    srcs = [
        "LLVMBaseTarget.cpp",
    ],
    hdrs = [
        "LLVMBaseTarget.h",
    ],
    deps = [
        ":LLVMIRPasses",
        ":LLVMTargetOptions",
        "//iree/compiler/Conversion/LinalgToLLVM",
        "//iree/compiler/Dialect/HAL/Target",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)

cc_library(
    name = "LLVMIR",
    srcs = [
        "LLVMIRTarget.cpp",
    ],
    hdrs = [
        "LLVMIRTarget.h",
    ],
    deps = [
        ":LLVMBase",
        ":LLVMTargetOptions",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/schemas:llvmir_executable_def_cc_fbs",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
        # TODO(ataei): Link with native target dep.
        "@llvm-project//llvm:x86_code_gen",
    ],
)

cc_library(
    name = "LLVMAOT",
    srcs = [
        "LLVMAOTTarget.cpp",
    ],
    hdrs = [
        "LLVMAOTTarget.h",
    ],
    deps = [
        ":LLVMBase",
        ":LLVMTargetOptions",
        ":TestLibrary",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/schemas:dylib_executable_def_cc_fbs",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
        # TODO(ataei): Link with native target dep.
        # "@llvm-project//llvm:x86_code_gen",
    ],
)

cc_embed_data(
    name = "TestLibrary",
    srcs = ["TestLibrary.dll"],
    cc_file_output = "TestLibrary.cpp",
    cpp_namespace = "mlir::iree_compiler",
    flatten = True,
    h_file_output = "TestLibrary.h",
)

cc_library(
    name = "LLVMIRPasses",
    srcs = [
        "LLVMIRPasses.cpp",
    ],
    hdrs = [
        "LLVMIRPasses.h",
    ],
    deps = [
        ":LLVMTargetOptions",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:passes",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LLVMTargetOptions",
    srcs = [
        "LLVMTargetOptions.cpp",
    ],
    hdrs = [
        "LLVMTargetOptions.h",
    ],
    deps = [
        "@llvm-project//llvm:passes",
        "@llvm-project//llvm:support",
    ],
)
