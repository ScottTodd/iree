// Copyright 2020 Google LLC
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

#include "iree/hal/dylib/dylib_executable.h"

#include <iostream>
#include <memory>

#include "flatbuffers/flatbuffers.h"
#include "iree/base/file_io.h"
#include "iree/hal/executable.h"
#include "iree/schemas/dylib_executable_def_generated.h"

namespace iree {
namespace hal {
namespace dylib {

// static
StatusOr<ref_ptr<DyLibExecutable>> DyLibExecutable::Load(
    hal::Allocator* allocator, ExecutableSpec spec, bool allow_aliasing_data) {
  auto module_def =
      ::flatbuffers::GetRoot<DyLibExecutableDef>(spec.executable_data.data());
  auto data =
      reinterpret_cast<const char*>(module_def->library_embedded()->data());
  const int size = module_def->library_embedded()->size();
  LOG(INFO) << "Loaded embedded library into memory, size: " << size;

  // Write the embedded library out to a temp file, since all of the dynamic
  // library APIs work with files. We could instead use in-memory files on
  // platforms where that is convenient.
  //
  // Names are selected to look like platform-specific dynamic library files
  // to increase the chances of opinionated dynamic library loaders finding
  // them:
  //   Windows: `C:\path\to\temp\dylib_executableXXXXXX.dll`
  //   Posix:   `/path/to/temp/libdylib_executableXXXXXX.so`
#if defined(IREE_PLATFORM_WINDOWS)
  std::string base_name = "dylib_executable";
#else
  std::string base_name = "libdylib_executable";
#endif
  ASSIGN_OR_RETURN(std::string temp_file, file_io::GetTempFile(base_name));
#if defined(IREE_PLATFORM_WINDOWS)
  temp_file += ".dll";
#else
  temp_file += ".so";
#endif

  absl::string_view data_view(data, size);
  RETURN_IF_ERROR(file_io::SetFileContents(temp_file, data_view));
  LOG(INFO) << "Wrote embedded library to " << temp_file;

  ASSIGN_OR_RETURN(auto executable_library,
                   DynamicLibrary::Load(temp_file.c_str()));
  LOG(INFO) << "Loaded library from temp file";

  auto times_two_fn = executable_library->GetSymbol<int (*)(int)>("times_two");
  LOG(INFO) << "Three times two is " << times_two_fn(3);

  return UnimplementedErrorBuilder(IREE_LOC) << "DyLibExecutable::Load NYI";

  // auto executable = make_ref<DyLibExecutable>(
  //     allocator, spec, allow_aliasing_data, std::move(executable_library));

  // return executable;
}

DyLibExecutable::DyLibExecutable(hal::Allocator* allocator, ExecutableSpec spec,
                                 bool allow_aliasing_data)
    : spec_(spec) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

DyLibExecutable::~DyLibExecutable() = default;

}  // namespace dylib
}  // namespace hal
}  // namespace iree
