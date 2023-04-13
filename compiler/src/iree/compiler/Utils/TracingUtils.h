// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TRACINGUTILS_H_
#define IREE_COMPILER_UTILS_TRACINGUTILS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"

//===----------------------------------------------------------------------===//
// IREE_COMPILER_TRACING_FEATURE_* flags and options
//===----------------------------------------------------------------------===//

// Enables IREE_COMPILER_TRACE_* macros for instrumented tracing.
#define IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION (1 << 0)

// Tracks all allocations (we know about) via new/delete/malloc/free.
#define IREE_COMPILER_TRACING_FEATURE_ALLOCATION_TRACKING (1 << 1)

// Captures callstacks up to IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH at all
// allocation events when allocation tracking is enabled.
#define IREE_COMPILER_TRACING_FEATURE_ALLOCATION_CALLSTACKS (1 << 2)

#if !defined(IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH)
#define IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH 16
#endif  // IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH

//===----------------------------------------------------------------------===//
// IREE_COMPILER_TRACING_MODE simple setting
//===----------------------------------------------------------------------===//
// TODO(scotttodd): Add C++ flags to control these at runtime if possible

// Set IREE_COMPILER_TRACING_FEATURES based on IREE_COMPILER_TRACING_MODE if
// the user hasn't overridden it with more specific settings.
//
// IREE_COMPILER_TRACING_MODE = 0: tracing disabled
// IREE_COMPILER_TRACING_MODE = 1: instrumentation
// IREE_COMPILER_TRACING_MODE = 2: same as 1 with added allocation tracking
// IREE_COMPILER_TRACING_MODE = 3: same as 2 with callstacks for allocations
#if !defined(IREE_COMPILER_TRACING_FEATURES)
#if defined(IREE_COMPILER_TRACING_MODE) && IREE_COMPILER_TRACING_MODE == 1
#define IREE_COMPILER_TRACING_FEATURES \
  (IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION)
#elif defined(IREE_COMPILER_TRACING_MODE) && IREE_COMPILER_TRACING_MODE == 2
#define IREE_COMPILER_TRACING_FEATURES             \
  (IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION | \
   IREE_COMPILER_TRACING_FEATURE_ALLOCATION_TRACKING)
#elif defined(IREE_COMPILER_TRACING_MODE) && IREE_COMPILER_TRACING_MODE == 3
#define IREE_COMPILER_TRACING_FEATURES                 \
  (IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION |     \
   IREE_COMPILER_TRACING_FEATURE_ALLOCATION_TRACKING | \
   IREE_COMPILER_TRACING_FEATURE_ALLOCATION_CALLSTACKS)
#else
#define IREE_COMPILER_TRACING_FEATURES 0
#endif  // IREE_COMPILER_TRACING_MODE
#endif  // !IREE_COMPILER_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// Tracy configuration
//===----------------------------------------------------------------------===//
// NOTE: order matters here as we are including files that require/define.

// Enable Tracy only when we are using tracing features.
#if IREE_COMPILER_TRACING_FEATURES != 0
#define TRACY_ENABLE 1
#endif  // IREE_COMPILER_TRACING_FEATURES

// Disable zone nesting verification in release builds.
// The verification makes it easy to find unbalanced zones but doubles the cost
// (at least) of each zone recorded. Run in debug builds to verify new
// instrumentation is correct before capturing traces in release builds.
#if defined(NDEBUG)
#define TRACY_NO_VERIFY 1
#endif  // NDEBUG

// Force callstack capture on all zones (even those without the C suffix).
#define TRACY_CALLSTACK 1

// Disable frame image capture to avoid the DXT compression code and the frame
// capture worker thread.
#define TRACY_NO_FRAME_IMAGE 1

// We don't care about vsync events as they can pollute traces and don't have
// much meaning in our workloads. If integrators still want them we can expose
// this as a tracing feature flag.
#define TRACY_NO_VSYNC_CAPTURE 1

// Flush the settings we have so far; settings after this point will be
// overriding values set by Tracy itself.
#if defined(TRACY_ENABLE)
#include "tracy/TracyC.h"  // IWYU pragma: export
#endif

//===----------------------------------------------------------------------===//
// C API used for Tracy control
//===----------------------------------------------------------------------===//
// These functions are implementation details and should not be called directly.
// Always use the macros (or C++ RAII types).

// Local zone ID used for the C IREE_COMPILER_TRACE_ZONE_* macros.
typedef uint32_t iree_compiler_zone_id_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_COMPILER_TRACING_FEATURES

typedef struct ___tracy_source_location_data iree_compiler_tracing_location_t;

#ifdef __cplusplus
#define iree_compiler_tracing_make_zone_ctx(zone_id) \
  TracyCZoneCtx { zone_id, 1 }
#else
#define iree_compiler_tracing_make_zone_ctx(zone_id) \
  (TracyCZoneCtx) { zone_id, 1 }
#endif  // __cplusplus

iree_compiler_zone_id_t iree_compiler_tracing_zone_begin_impl(
    const iree_compiler_tracing_location_t *src_loc, const char *name,
    size_t name_length);
iree_compiler_zone_id_t iree_compiler_tracing_zone_begin_external_impl(
    const char *file_name, size_t file_name_length, uint32_t line,
    const char *function_name, size_t function_name_length, const char *name,
    size_t name_length);
void iree_compiler_tracing_zone_end(iree_compiler_zone_id_t zone_id);

#endif  // IREE_COMPILER_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instrumentation macros (C)
//===----------------------------------------------------------------------===//

// Colors used for messages based on the level provided to the macro.
enum {
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_ERROR = 0xFF0000u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_WARNING = 0xFFFF00u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_INFO = 0xFFFFFFu,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_VERBOSE = 0xC0C0C0u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_DEBUG = 0x00FF00u,
};

#if IREE_COMPILER_TRACING_FEATURES & \
    IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION

// Begins a new zone with the parent function name.
#define IREE_COMPILER_TRACE_ZONE_BEGIN(zone_id) \
  IREE_COMPILER_TRACE_ZONE_BEGIN_NAMED(zone_id, NULL)

// Begins a new zone with the given compile-time literal name.
#define IREE_COMPILER_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)           \
  static const iree_compiler_tracing_location_t TracyConcat(                  \
      __tracy_source_location, __LINE__) = {name_literal, __FUNCTION__,       \
                                            __FILE__, (uint32_t)__LINE__, 0}; \
  iree_compiler_zone_id_t zone_id = iree_compiler_tracing_zone_begin_impl(    \
      &TracyConcat(__tracy_source_location, __LINE__), NULL, 0);

// Begins a new zone with the given runtime dynamic string name.
// The |value| string will be copied into the trace buffer.
#define IREE_COMPILER_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name,        \
                                                     name_length)          \
  static const iree_compiler_tracing_location_t TracyConcat(               \
      __tracy_source_location, __LINE__) = {0, __FUNCTION__, __FILE__,     \
                                            (uint32_t)__LINE__, 0};        \
  iree_compiler_zone_id_t zone_id = iree_compiler_tracing_zone_begin_impl( \
      &TracyConcat(__tracy_source_location, __LINE__), (name), (name_length));

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_COMPILER_TRACE_ZONE_BEGIN_EXTERNAL(               \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                   \
  iree_compiler_zone_id_t zone_id =                            \
      iree_compiler_tracing_zone_begin_external_impl(          \
          file_name, file_name_length, line, function_name,    \
          function_name_length, name, name_length)

// Sets the dynamic color of the zone to an XXBBGGRR value.
#define IREE_COMPILER_TRACE_ZONE_SET_COLOR(zone_id, color_xbgr)          \
  ___tracy_emit_zone_color(iree_compiler_tracing_make_zone_ctx(zone_id), \
                           color_xbgr);

// Appends an integer value to the parent zone. May be called multiple times.
#define IREE_COMPILER_TRACE_ZONE_APPEND_VALUE(zone_id, value) \
  ___tracy_emit_zone_value(iree_compiler_tracing_make_zone_ctx(zone_id), value);

// Appends a string value to the parent zone. May be called multiple times.
// The |value| string will be copied into the trace buffer.
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT(...)                     \
  IREE_COMPILER_TRACE_IMPL_GET_VARIADIC_(                             \
      (__VA_ARGS__, IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_STRING_VIEW, \
       IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_CSTRING))                 \
  (__VA_ARGS__)
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value) \
  IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value,   \
                                                   strlen(value))
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value,       \
                                                         value_length)         \
  ___tracy_emit_zone_text(iree_compiler_tracing_make_zone_ctx(zone_id), value, \
                          value_length)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_COMPILER_TRACE_ZONE_END(zone_id) \
  iree_compiler_tracing_zone_end(zone_id)

// Demarcates an advancement of the top-level unnamed frame group.
#define IREE_COMPILER_TRACE_FRAME_MARK() ___tracy_emit_frame_mark(NULL)
// Demarcates an advancement of a named frame group.
#define IREE_COMPILER_TRACE_FRAME_MARK_NAMED(name_literal) \
  ___tracy_emit_frame_mark(name_literal)
// Begins a discontinuous frame in a named frame group.
// Must be properly matched with a IREE_COMPILER_TRACE_FRAME_MARK_NAMED_END.
#define IREE_COMPILER_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal) \
  ___tracy_emit_frame_mark_start(name_literal)
// Ends a discontinuous frame in a named frame group.
#define IREE_COMPILER_TRACE_FRAME_MARK_END_NAMED(name_literal) \
  ___tracy_emit_frame_mark_end(name_literal)

// Logs a message at the given logging level to the trace.
// The message text must be a compile-time string literal.
#define IREE_COMPILER_TRACE_MESSAGE(level, value_literal) \
  ___tracy_emit_messageLC(value_literal, IREE_TRACING_MESSAGE_LEVEL_##level, 0)
// Logs a dynamically-allocated message at the given logging level to the trace.
// The string |value| will be copied into the trace buffer.
#define IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(level, value_string)   \
  ___tracy_emit_messageC(value_string.data(), value_string.size(), \
                         IREE_TRACING_COMPILER_MESSAGE_LEVEL_##level, 0)

// Utilities:
#define IREE_COMPILER_TRACE_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) \
  NAME
#define IREE_COMPILER_TRACE_IMPL_GET_VARIADIC_(args) \
  IREE_COMPILER_TRACE_IMPL_GET_VARIADIC_HELPER_ args

#else
#define IREE_COMPILER_TRACE_ZONE_BEGIN(zone_id) \
  iree_compiler_zone_id_t zone_id = 0;          \
  (void)zone_id;
#define IREE_COMPILER_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_COMPILER_TRACE_ZONE_BEGIN(zone_id)
#define IREE_COMPILER_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, \
                                                     name_length)   \
  IREE_COMPILER_TRACE_ZONE_BEGIN(zone_id)
#define IREE_COMPILER_TRACE_ZONE_BEGIN_EXTERNAL(               \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                   \
  IREE_COMPILER_TRACE_ZONE_BEGIN(zone_id)
#define IREE_COMPILER_TRACE_ZONE_SET_COLOR(zone_id, color_xrgb)
#define IREE_COMPILER_TRACE_ZONE_APPEND_VALUE(zone_id, value)
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT(zone_id, ...)
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value)
#define IREE_COMPILER_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, \
                                                         value_length)
#define IREE_COMPILER_TRACE_ZONE_END(zone_id)
#define IREE_COMPILER_TRACE_FRAME_MARK()
#define IREE_COMPILER_TRACE_FRAME_MARK_NAMED(name_literal)
#define IREE_COMPILER_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
#define IREE_COMPILER_TRACE_FRAME_MARK_END_NAMED(name_literal)
#define IREE_COMPILER_TRACE_MESSAGE(level, value_literal)
#define IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(level, value_string)
#endif  // IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Allocation tracking macros (C/C++)
//===----------------------------------------------------------------------===//
//
// IREE_COMPILER_TRACE_ALLOC: records an malloc.
// IREE_COMPILER_TRACE_FREE: records a free.
//
// NOTE: realloc must be recorded as a FREE/ALLOC pair.

#if IREE_COMPILER_TRACING_FEATURES & \
    IREE_COMPILER_TRACING_FEATURE_ALLOCATION_TRACKING

#if IREE_COMPILER_TRACING_FEATURES & \
    IREE_COMPILER_TRACING_FEATURE_ALLOCATION_CALLSTACKS

#define IREE_COMPILER_TRACE_ALLOC(ptr, size) \
  ___tracy_emit_memory_alloc_callstack(      \
      ptr, size, IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH, 0)
#define IREE_COMPILER_TRACE_FREE(ptr)  \
  ___tracy_emit_memory_free_callstack( \
      ptr, IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH, 0)
#define IREE_COMPILER_TRACE_ALLOC_NAMED(name, ptr, size) \
  ___tracy_emit_memory_alloc_callstack_named(            \
      ptr, size, IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH, 0, name)
#define IREE_COMPILER_TRACE_FREE_NAMED(name, ptr) \
  ___tracy_emit_memory_free_callstack_named(      \
      ptr, IREE_COMPILER_TRACING_MAX_CALLSTACK_DEPTH, 0, name)

#else

#define IREE_COMPILER_TRACE_ALLOC(ptr, size) \
  ___tracy_emit_memory_alloc(ptr, size, 0)
#define IREE_COMPILER_TRACE_FREE(ptr) ___tracy_emit_memory_free(ptr, 0)
#define IREE_COMPILER_TRACE_ALLOC_NAMED(name, ptr, size) \
  ___tracy_emit_memory_alloc_named(ptr, size, 0, name)
#define IREE_COMPILER_TRACE_FREE_NAMED(name, ptr) \
  ___tracy_emit_memory_free_named(ptr, 0, name)

#endif  // IREE_COMPILER_TRACING_FEATURE_ALLOCATION_CALLSTACKS

#else
#define IREE_COMPILER_TRACE_ALLOC(ptr, size)
#define IREE_COMPILER_TRACE_FREE(ptr)
#define IREE_COMPILER_TRACE_ALLOC_NAMED(name, ptr, size)
#define IREE_COMPILER_TRACE_FREE_NAMED(name, ptr)
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// PassTracing (PassInstrumentation)
//===----------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

// Instruments passes using IREE's runtime tracing support.
//
// Usage:
//   passManager.addInstrumentation(std::make_unique<PassTracing>());
struct PassTracing : public PassInstrumentation {
  PassTracing() {}
  ~PassTracing() override = default;

#if IREE_COMPILER_TRACING_FEATURES & \
    IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;
#endif  // IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION
};

//===----------------------------------------------------------------------===//
// MarkBeginPass / MarkEndPass
//===----------------------------------------------------------------------===//

#if IREE_COMPILER_TRACING_FEATURES & \
    IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION

// Adds a pass to |passManager| that marks the beginning of a named frame.
// * |frameName| must be a null-terminated string
// * |frameName| must use the same underlying storage as the name used with
//   IREE_COMPILER_TRACE_ADD_END_FRAME_PASS
#define IREE_COMPILER_TRACE_ADD_BEGIN_FRAME_PASS(passManager, frameName) \
  passManager.addPass(createTraceFrameMarkBeginPass(frameName));

// Adds a pass to |passManager| that marks the end of a named frame.
// * |frameName| must be a null-terminated string
// * |frameName| must use the same underlying storage as the name used with
//   IREE_COMPILER_TRACE_ADD_BEGIN_FRAME_PASS
#define IREE_COMPILER_TRACE_ADD_END_FRAME_PASS(passManager, frameName) \
  passManager.addPass(createTraceFrameMarkEndPass(frameName));

std::unique_ptr<OperationPass<mlir::ModuleOp>> createTraceFrameMarkBeginPass(
    llvm::StringRef frameName = "");
std::unique_ptr<OperationPass<mlir::ModuleOp>> createTraceFrameMarkEndPass(
    llvm::StringRef frameName = "");

#else
#define IREE_COMPILER_TRACE_ADD_BEGIN_FRAME_PASS(passManager, frameName)
#define IREE_COMPILER_TRACE_ADD_END_FRAME_PASS(passManager, frameName)
#endif  // IREE_COMPILER_TRACING_FEATURE_INSTRUMENTATION

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TRACINGUTILS_H_
