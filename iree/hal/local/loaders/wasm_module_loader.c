// Copyright 2021 Google LLC
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

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/local_executable.h"
#include "third_party/wasmtime/include/wasi.h"
#include "third_party/wasmtime/include/wasm.h"
#include "third_party/wasmtime/include/wasmtime.h"

//===----------------------------------------------------------------------===//
// iree_hal_wasm_executable_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_local_executable_t base;

  // Loaded ELF module.
  // iree_elf_module_t module;

  // Name used for the file field in tracy and debuggers.
  // iree_string_view_t identifier;

  // wasi_instance_t* wasi_instance;
  // wasmtime_linker_t* wasmtime_linker;

  wasm_module_t* wasm_module;
  wasm_instance_t* wasm_instance;

  // Queried metadata from the library.
  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;
} iree_hal_wasm_executable_t;

extern const iree_hal_local_executable_vtable_t iree_hal_wasm_executable_vtable;

static iree_status_t iree_hal_wasm_executable_query_library(
    iree_hal_wasm_executable_t* executable) {
  // TODO(scotttodd): call the IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME function

  // // Get the exported symbol used to get the library metadata.
  // iree_hal_executable_library_query_fn_t query_fn = NULL;
  // IREE_RETURN_IF_ERROR(iree_elf_module_lookup_export(
  //     &executable->module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME,
  //     (void**)&query_fn));

  // // Query for a compatible version of the library.
  // executable->library.header =
  //     (const iree_hal_executable_library_header_t**)iree_elf_call_p_ip(
  //         query_fn, IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION,
  //         /*reserved=*/NULL);
  // if (!executable->library.header) {
  //   return iree_make_status(
  //       IREE_STATUS_FAILED_PRECONDITION,
  //       "executable does not support this version of the runtime (%d)",
  //       IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION);
  // }
  // const iree_hal_executable_library_header_t* header =
  //     *executable->library.header;

  // // Ensure that if the library is built for a particular sanitizer that we
  // also
  // // were compiled with that sanitizer enabled.
  // switch (header->sanitizer) {
  //   case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE:
  //     // Always safe even if the host has a sanitizer enabled; it just means
  //     // that we won't be able to catch anything from within the executable,
  //     // however checks outside will (often) still trigger when guard pages
  //     are
  //     // dirtied/etc.
  //     break;
  //   default:
  //     return iree_make_status(IREE_STATUS_UNAVAILABLE,
  //                             "executable requires sanitizer but they are not
  //                             " "yet supported with embedded libraries: %u",
  //                             (uint32_t)header->sanitizer);
  // }

  // executable->identifier = iree_make_cstring_view(header->name);

  return iree_ok_status();
}

static iree_status_t iree_hal_wasm_executable_create(
    wasm_engine_t* wasm_engine, wasm_store_t* wasm_store,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t wasm_module_data,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(wasm_module_data.data && wasm_module_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_wasm_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)(((uint8_t*)executable) +
                                               sizeof(*executable));
    iree_hal_local_executable_initialize(
        &iree_hal_wasm_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
  }

  // TODO(scotttodd): move to function
  if (iree_status_is_ok(status)) {
    wasm_byte_vec_t wasm_module_bytes;
    // TODO(scotttodd): fix `warning C4090: '=': different 'const' qualifiers`
    wasm_module_bytes.data = wasm_module_data.data;
    wasm_module_bytes.size = wasm_module_data.data_length;

    executable->wasm_module = NULL;
    wasmtime_error_t* wasm_error = wasmtime_module_new(
        wasm_engine, &wasm_module_bytes, &executable->wasm_module);
    if (!executable->wasm_module) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "wasmtime_module_new error: '%s'", wasm_error);
    }
  }

  // if (iree_status_is_ok(status)) {
  //   wasi_config_t* wasi_config = wasi_config_new();
  //   // TODO(scotttodd): error handling (check not NULL)
  //   wasi_config_inherit_argv(wasi_config);
  //   wasi_config_inherit_env(wasi_config);
  //   wasi_config_inherit_stdin(wasi_config);
  //   wasi_config_inherit_stdout(wasi_config);
  //   wasi_config_inherit_stderr(wasi_config);
  //   wasm_trap_t* wasm_trap = NULL;
  //   executable->wasi_instance = wasi_instance_new(
  //       wasm_store, "wasi_snapshot_preview1", wasi_config, &wasm_trap);
  //   if (!executable->wasi_instance) {
  //     status =
  //         iree_make_status(IREE_STATUS_INTERNAL, "wasi_instance_new failed");
  //   }
  // }

  // if (iree_status_is_ok(status)) {
  //   executable->wasmtime_linker = wasmtime_linker_new(wasm_store);
  //   wasmtime_error_t* wasm_error = wasmtime_linker_define_wasi(
  //       executable->wasmtime_linker, executable->wasi_instance);
  //   if (wasm_error) {
  //     status = iree_make_status(IREE_STATUS_INTERNAL,
  //                               "wasmtime_module_new error: '%s'",
  //                               wasm_error);
  //   }
  // }

  if (iree_status_is_ok(status)) {
    executable->wasm_instance = NULL;
    wasm_trap_t* trap = NULL;
    wasm_extern_vec_t imports = WASM_EMPTY_VEC;
    wasmtime_error_t* wasm_error =
        wasmtime_instance_new(wasm_store, executable->wasm_module, &imports,
                              &executable->wasm_instance, &trap);
    if (!executable->wasm_instance || wasm_error) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "wasmtime_instance_new error: '%s'", wasm_error);
    }
  }

  if (iree_status_is_ok(status)) {
    wasm_exporttype_vec_t module_exports;
    wasm_module_exports(executable->wasm_module, &module_exports);

    for (int i = 0; i < module_exports.size; ++i) {
      const wasm_name_t* export_name =
          wasm_exporttype_name(module_exports.data[i]);
      status = iree_ok_status();
    }

    wasm_extern_vec_t instance_externs;
    wasm_instance_exports(executable->wasm_instance, &instance_externs);

    // (export "memory" (memory 0))
    // (export "iree_hal_executable_library_query" (func
    // $iree_hal_executable_library_query))

    for (int i = 0; i < instance_externs.size; ++i) {
      wasm_externkind_t externkind = wasm_extern_kind(instance_externs.data[i]);
      switch (externkind) {
        case WASM_EXTERN_FUNC: {
          status = iree_ok_status();
          // wasm_functype_t* functype =
          // wasm_externtype_as_functype(externtype);
          break;
        }
        case WASM_EXTERN_GLOBAL:
          status = iree_ok_status();
          break;
        case WASM_EXTERN_TABLE:
          status = iree_ok_status();
          break;
        case WASM_EXTERN_MEMORY:
          status = iree_ok_status();
          break;
        default:
          break;
      }
      // const wasm_name_t* externname = wasm_exporttype_
    }

    // index 0 export is memory
    // index 1 export is 'iree_hal_executable_library_query' function
    wasm_func_t* query_func = wasm_extern_as_func(instance_externs.data[1]);

    // TODO(scotttodd): call query_func, try to call `multiply_dispatch_0`
    //
    // * maybe `multiply_dispatch_0` should be exported too?
    //   queryLibraryFunc is the only exported function
    //   try passing entryPointOp functions to configureModule
    // * wasmtime_funcref_table_get -> wasm_func_t
  }

  if (iree_status_is_ok(status)) {
    // Attempt to load the ELF module.
    // status = iree_elf_module_initialize_from_memory(
    //     wasm_module_data, /*import_table=*/NULL, host_allocator,
    //     &executable->module);
  }
  if (iree_status_is_ok(status)) {
    // Query metadata and get the entry point function pointers.
    // status = iree_hal_wasm_executable_query_library(executable);
  }
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(
          caching_mode,
          IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION)) {
    // // Check to make sure that the entry point count matches the layouts
    // // provided.
    // if (executable->library.v0->entry_point_count != executable_layout_count)
    // {
    //   return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
    //                           "executable provides %u entry points but caller
    //                           " "provided %zu; must match",
    //                           executable->library.v0->entry_point_count,
    //                           executable_layout_count);
    // }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_wasm_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_wasm_executable_t* executable =
      (iree_hal_wasm_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // iree_elf_module_deinitialize(&executable->module);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_wasm_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id) {
  iree_hal_wasm_executable_t* executable =
      (iree_hal_wasm_executable_t*)base_executable;
  //   const iree_hal_executable_library_v0_t* library = executable->library.v0;

  //   if (IREE_UNLIKELY(ordinal >= library->entry_point_count)) {
  //     return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
  //                             "entry point ordinal out of bounds");
  //   }

  // #if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  //   iree_string_view_t entry_point_name = iree_string_view_empty();
  //   if (library->entry_point_names != NULL) {
  //     entry_point_name =
  //         iree_make_cstring_view(library->entry_point_names[ordinal]);
  //   }
  //   if (iree_string_view_is_empty(entry_point_name)) {
  //     entry_point_name = iree_make_cstring_view("unknown_elf_call");
  //   }
  //   IREE_TRACE_ZONE_BEGIN_EXTERNAL(
  //       z0, executable->identifier.data, executable->identifier.size,
  //       ordinal, entry_point_name.data, entry_point_name.size, NULL, 0);
  // #endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  //   int ret = iree_elf_call_i_pp(library->entry_points[ordinal],
  //                                (void*)dispatch_state, (void*)workgroup_id);

  //   IREE_TRACE_ZONE_END(z0);

  //   return ret == 0 ? iree_ok_status()
  //                   : iree_make_status(
  //                         IREE_STATUS_INTERNAL,
  //                         "executable entry point returned catastrophic error
  //                         %d", ret);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "iree_hal_wasm_executable_issue_call NYI");
}

const iree_hal_local_executable_vtable_t iree_hal_wasm_executable_vtable = {
    .base =
        {
            .destroy = iree_hal_wasm_executable_destroy,
        },
    .issue_call = iree_hal_wasm_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_wasm_module_loader_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;

  // TODO(scotttodd): move to per-device
  //     or store on iree_hal_allocator_t and create one of those per device?
  wasm_engine_t* wasm_engine;
  wasm_store_t* wasm_store;
} iree_hal_wasm_module_loader_t;

extern const iree_hal_executable_loader_vtable_t
    iree_hal_wasm_module_loader_vtable;

iree_status_t iree_hal_wasm_module_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_wasm_module_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(&iree_hal_wasm_module_loader_vtable,
                                          &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  if (iree_status_is_ok(status)) {
    executable_loader->wasm_engine = wasm_engine_new();
    if (!executable_loader->wasm_engine) {
      status = iree_make_status(IREE_STATUS_INTERNAL, "wasm_engine_new failed");
    }
  }
  if (iree_status_is_ok(status)) {
    executable_loader->wasm_store =
        wasm_store_new(executable_loader->wasm_engine);
    if (!executable_loader->wasm_store) {
      status = iree_make_status(IREE_STATUS_INTERNAL, "wasm_store_new failed");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_wasm_module_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_wasm_module_loader_t* executable_loader =
      (iree_hal_wasm_module_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable_loader->wasm_store) {
    wasm_store_delete(executable_loader->wasm_store);
  }
  if (executable_loader->wasm_engine) {
    wasm_engine_delete(executable_loader->wasm_engine);
  }

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_wasm_module_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("Wasm"));
}

static iree_status_t iree_hal_wasm_module_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_wasm_module_loader_t* executable_loader =
      (iree_hal_wasm_module_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load of the wasm module and wrap it in an executable handle.
  iree_status_t status = iree_hal_wasm_executable_create(
      executable_loader->wasm_engine, executable_loader->wasm_store,
      executable_spec->caching_mode, executable_spec->executable_data,
      executable_spec->executable_layout_count,
      executable_spec->executable_layouts, executable_loader->host_allocator,
      out_executable);

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "iree_hal_wasm_module_loader_try_load NYI");
  // return status;
}

const iree_hal_executable_loader_vtable_t iree_hal_wasm_module_loader_vtable = {
    .destroy = iree_hal_wasm_module_loader_destroy,
    .query_support = iree_hal_wasm_module_loader_query_support,
    .try_load = iree_hal_wasm_module_loader_try_load,
};
