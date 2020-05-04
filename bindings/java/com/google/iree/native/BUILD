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

cc_library(
    name = "native_deps",
    srcs = [
        "context_jni.cc",
    ],
    tags = [
        "noga",
    ],
    deps = [
        # "@bazel_tools//tools/jdk:jni",
        "//iree/base:logging",
        # TODO(jennik): Update this for OSS side.
    ],
    alwayslink = 1,
    linkopts = ["-ldl", "-lm"],
)

cc_binary(
    name = "libiree_native.so",
    deps = [
        ":native_deps",
    ],
    linkshared = 1,
)

# This shouldn't be necessary
cc_library(
    name = "iree_native_so",
    deps = [":libiree_native.so"]
)
