# Copyright 2021 Google LLC
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

set(WASMTIME_ROOT "${IREE_ROOT_DIR}/third_party/wasmtime/")

# TODO(scotttodd): move config from iree\hal\local\loaders\CMakeLists.txt here

external_cc_library(
  PACKAGE
    wasmtime
  NAME
    wasmtime_c
  ROOT
    ${WASMTIME_ROOT}
  # HDRS
  #   "include/wasi.h"
  #   "include/wasm.h"
  #   "include/wasmtime.h"
  # SRCS
  #   "lib/wasmtime.lib"
  INCLUDES
    "${PROJECT_SOURCE_DIR}/third_party/wasmtime/include"
  # LINKOPTS
  #   "/LIB:${PROJECT_SOURCE_DIR}/third_party/wasmtime/lib/wasmtime.lib"
)
