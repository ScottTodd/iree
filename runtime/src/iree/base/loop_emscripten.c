// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// #include "iree/base/internal/math.h"
// #include "iree/base/internal/wait_handle.h"
#include "iree/base/loop_emscripten.h"

#include <emscripten.h>

#include "iree/base/assert.h"

extern void loop_emscripten_log_2();
extern iree_status_t loopCommandCall(iree_loop_callback_fn_t callback,
                                     void* user_data, iree_loop_t loop);

//===----------------------------------------------------------------------===//
// iree_loop_emscripten_t
//===----------------------------------------------------------------------===//

typedef struct iree_loop_emscripten_t {
  iree_allocator_t allocator;

  // TODO(scotttodd): other data

  //   iree_loop_run_ring_t* run_ring;
  //   iree_loop_wait_list_t* wait_list;

  // Trailing data:
  // + iree_loop_run_ring_storage_size
  // + iree_loop_wait_list_storage_size
} iree_loop_emscripten_t;

IREE_API_EXPORT iree_status_t iree_loop_emscripten_allocate(
    iree_allocator_t allocator, iree_loop_emscripten_t** out_loop_emscripten) {
  IREE_ASSERT_ARGUMENT(out_loop_emscripten);

  loop_emscripten_log_2();

  const iree_host_size_t loop_emscripten_size =
      iree_host_align(sizeof(iree_loop_emscripten_t), iree_max_align_t);

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, loop_emscripten_size, (void**)&storage));
  iree_loop_emscripten_t* loop_emscripten = (iree_loop_emscripten_t*)storage;
  loop_emscripten->allocator = allocator;

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_loop_emscripten = loop_emscripten;
  } else {
    iree_loop_emscripten_free(loop_emscripten);
  }
  return status;
}

IREE_API_EXPORT void iree_loop_emscripten_free(
    iree_loop_emscripten_t* loop_emscripten) {
  IREE_ASSERT_ARGUMENT(loop_emscripten);
  iree_allocator_t allocator = loop_emscripten->allocator;

  // Abort all pending operations.
  // This will issue callbacks for each operation that was aborted directly
  // with IREE_STATUS_ABORTED.
  // To ensure we don't enqueue more work while aborting we NULL out the lists.
  //   iree_loop_run_ring_t* run_ring = loop_emscripten->run_ring;
  //   iree_loop_wait_list_t* wait_list = loop_emscripten->wait_list;
  //   loop_emscripten->run_ring = NULL;
  //   loop_emscripten->wait_list = NULL;
  //   iree_loop_wait_list_abort_all(wait_list);
  //   iree_loop_run_ring_abort_all(run_ring);

  // After all operations are cleared we can release the data structures.
  //   iree_loop_run_ring_deinitialize(run_ring);
  //   iree_loop_wait_list_deinitialize(wait_list);
  iree_allocator_free(allocator, loop_emscripten);
}

static void iree_loop_emscripten_run_call(
    iree_loop_emscripten_t* loop_emscripten, iree_loop_t loop,
    const iree_loop_call_params_t params, iree_status_t op_status) {
  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, op_status);
  if (!iree_status_is_ok(status)) {
    // iree_loop_emscripten_emit_error(loop, status);
  }
}

static void iree_loop_emscripten_run_dispatch(
    iree_loop_emscripten_t* loop_emscripten, iree_loop_t loop,
    const iree_loop_dispatch_params_t params) {
  iree_status_t status = iree_ok_status();

  // We run all workgroups before issuing the completion callback.
  // If any workgroup fails we exit early and pass the failing status back to
  // the completion handler exactly once.
  uint32_t workgroup_count_x = params.workgroup_count_xyz[0];
  uint32_t workgroup_count_y = params.workgroup_count_xyz[1];
  uint32_t workgroup_count_z = params.workgroup_count_xyz[2];
  iree_status_t workgroup_status = iree_ok_status();
  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_status =
            params.workgroup_fn(params.callback.user_data, loop, x, y, z);
        if (!iree_status_is_ok(workgroup_status)) goto workgroup_failed;
      }
    }
  }
workgroup_failed:

  // Fire the completion callback with either success or the first error hit by
  // a workgroup.
  status =
      params.callback.fn(params.callback.user_data, loop, workgroup_status);
  if (!iree_status_is_ok(status)) {
    // iree_loop_emscripten_emit_error(loop, status);
  }
}

// // Drains work from the loop until all work in |scope| has completed.
// // A NULL |scope| indicates all work from all scopes should be drained.
// static iree_status_t iree_loop_emscripten_drain_scope(
//     iree_loop_emscripten_t* loop_emscripten,
//     iree_loop_emscripten_scope_t* scope, iree_time_t deadline_ns) {
//   do {
//     // If we are draining a particular scope we can bail whenever there's no
//     // more work remaining.
//     if (scope && !scope->pending_count) break;

//     // Run an op from the runnable queue.
//     // We dequeue operations here so that re-entrant enqueuing works.
//     // We only want to run one op at a time before checking our deadline so
//     that
//     // we don't get into infinite loops or exceed the deadline (too much).
//     iree_loop_run_op_t run_op;
//     if (iree_loop_run_ring_dequeue(loop_emscripten->run_ring, &run_op)) {
//       iree_loop_t loop = {
//           .self = run_op.scope,
//           .ctl = iree_loop_emscripten_ctl,
//       };
//       switch (run_op.command) {
//         case IREE_LOOP_COMMAND_CALL:
//           iree_loop_emscripten_run_call(loop_emscripten, loop,
//                                         run_op.params.call, run_op.status);
//           break;
//         case IREE_LOOP_COMMAND_DISPATCH:
//           iree_loop_emscripten_run_dispatch(loop_emscripten, loop,
//                                             run_op.params.dispatch);
//           break;
//       }
//       continue;  // loop back around only if under the deadline
//     }

//     // -- if here then the run ring is currently empty --

//     // If there are no pending waits then the drain has completed.
//     if (iree_loop_wait_list_is_empty(loop_emscripten->wait_list)) {
//       break;
//     }

//     // Scan the wait list and check for resolved ops.
//     // If there are any waiting ops the next earliest timeout is returned. An
//     // immediate timeout indicates that there's work in the run ring and we
//     // shouldn't perform a wait operation this go around the loop.
//     iree_time_t earliest_deadline_ns = IREE_TIME_INFINITE_FUTURE;
//     IREE_RETURN_AND_END_ZONE_IF_ERROR(
//         z0, iree_loop_wait_list_scan(loop_emscripten->wait_list,
//                                      loop_emscripten->run_ring,
//                                      &earliest_deadline_ns));
//     if (earliest_deadline_ns != IREE_TIME_INFINITE_PAST &&
//         earliest_deadline_ns != IREE_TIME_INFINITE_FUTURE) {
//       // Commit the wait operation, waiting up until the minimum of the user
//       // specified and wait list derived values.
//       iree_time_t wait_deadline_ns = earliest_deadline_ns < deadline_ns
//                                          ? earliest_deadline_ns
//                                          : deadline_ns;
//       IREE_RETURN_AND_END_ZONE_IF_ERROR(
//           z0, iree_loop_wait_list_commit(loop_emscripten->wait_list,
//                                          loop_emscripten->run_ring,
//                                          wait_deadline_ns));
//     }
//   } while (iree_time_now() < deadline_ns);

//   return iree_ok_status();
// }

IREE_API_EXPORT iree_status_t iree_loop_emscripten_wait_idle(
    iree_loop_emscripten_t* loop_emscripten, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(loop_emscripten);
  //   iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  //   iree_status_t status = iree_loop_emscripten_drain_scope(
  //       loop_emscripten, /*scope=*/NULL, deadline_ns);
  //   return status;
  return iree_ok_status();
}

// TODO(scotttodd): wait sources:
//     IREE_LOOP_COMMAND_WAIT_ONE
//     IREE_LOOP_COMMAND_WAIT_ANY
//     IREE_LOOP_COMMAND_WAIT_ALL
//   wait source backed by a Promise
//   wait source import from futex to Promise
//   Promise.then()
//   create semaphore (webgpu nop_semaphore changes, promise_semaphore?)

// TODO(scotttodd): move this JS code out into a library, with
//    * an object for storing state
//    * separate functions for each command case

EM_JS(void, test_function, (void* function_to_call), {
  console.log('iree_loop_emscripten test_function');
  console.log('function_to_call:', function_to_call);

  // Call the provided function pointer with a single integer argument.
  Module['dynCall_vi'](function_to_call, 123);
});

void test_function_pointer_from_js(int arg) {
  fprintf(stdout, "test_function_pointer_from_js, arg: %d\n", arg);
}

// static iree_status_t iree_loop_emscripten_enqueue(
//     iree_loop_emscripten_t* loop_emscripten, void* data) {
//   return iree_ok_status();
// }

// EM_JS_DEPS(iree_loop_emscripten_ctl, "$dynCall");

// clang-format off
EM_JS(void, loop_ctl, (int command, const void* params, void* inout_ptr), {
  console.log('iree_loop_emscripten loop_ctl');
  console.log('Module: ', Module);
  // console.log('Module.Runtime.dynCall: ', Module.Runtime.dynCall);
  // console.log('dynCall: ', dynCall);
  console.log('command: ', command);
  console.log('params: ', params);
  console.log('inout_ptr: ', inout_ptr);
  // console.log('arg1: ', $1);

  // TODO(scotttodd): extract params struct
  // if IREE_LOOP_COMMAND_CALL:
  //   iree_loop_call_params_t
  //     callback
  //       fn
  //       user_data
  //     priority

  // assume parmas is iree_loop_call_params_t
  // TODO(scotttodd): unpack in C, just take Number args here with no HEAPU32
  const call_callback_fn = HEAPU32[(params + 0) >> 2];
  const call_callback_user_data = HEAPU32[(params + 4) >> 2];
  const call_priority = HEAPU32[(params + 8) >> 2];

  console.log('call_callback_fn:', call_callback_fn);
  console.log('call_callback_user_data:', call_callback_user_data);
  console.log('call_priority:', call_priority);

  const ret = Module['dynCall_iiii'](call_callback_fn, call_callback_user_data,
                                     /*loop=*/0, /*status=*/0);
  console.log('ret:', ret);

  setTimeout(() => {
      console.log('loop_ctl -> setImmediate');
      // iree_status_t status =
      //     params.callback.fn(params.callback.user_data, loop,
      //     iree_ok_status());
  }, 0);
});
// clang-format on

// Control function for the Emscripten loop.
// DO NOT SUBMIT
// ???? |self| must be an iree_loop_emscripten_scope_t ????.
IREE_API_EXPORT iree_status_t
iree_loop_emscripten_ctl(void* self, iree_loop_command_t command,
                         const void* params, void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(self);

  test_function(test_function_pointer_from_js);

  // iree_loop_emscripten_t* loop_emscripten = (iree_loop_emscripten_t*)self;

  // iree_loop_call_params_t* call_params = (iree_loop_call_params_t*)params;

  // TODO(scotttodd): pass the entire command/params/etc. to JS?
  // loop_ctl(command, params, inout_ptr);

  iree_loop_call_params_t* call_params = (iree_loop_call_params_t*)params;
  iree_status_t status =
      loopCommandCall(call_params->callback.fn, call_params->callback.user_data,
                      iree_loop_null());
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "loopCommandCall was successful\n");
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  // clang-format off
  // EM_ASM({
  //   console.log('iree_loop_emscripten_ctl');
  //   console.log('Module: ', Module);
  //   // console.log('Module.Runtime.dynCall: ', Module.Runtime.dynCall);
  //   // console.log('dynCall: ', dynCall);
  //   console.log('arg0: ', $0);
  //   // console.log('arg1: ', $1);

  //   // Call the provided function pointer with a single integer argument.
  //   // Module.dynCall_viii()
  //   Module['dynCall_vi']($0, 123);

  //   setTimeout(() => {
  //     //
  //     console.log('iree_loop_emscripten_ctl -> setImmediate');

  //     //
  //     // iree_status_t status =
  //     //     params.callback.fn(params.callback.user_data, loop, iree_ok_status());
  //   }, 0);
  // }, test_function_pointer_from_js);
  // }, call_params->callback.fn, call_params->callback.user_data);
  // clang-format on

  // TODO(scotttodd): wrap args in void* struct?
  //     allocator alloc, hop to js, call c, allocator free

  // TODO(scotttodd): pass loop for reentrant scheduling
  // iree_status_t status = call_params->callback.fn(
  //     call_params->callback.user_data, iree_loop_null(), iree_ok_status());
  // if (iree_status_is_ok(status)) {
  //   fprintf(stdout, "callback fn was successful\n");
  // } else {
  //   iree_status_fprint(stderr, status);
  //   iree_status_free(status);
  // }

  // Original C++:
  //     callback.fn(callback.user_data);
  // With setTimeout:
  //     EM_ASM({
  //       setTimeout(() => {
  //         Module['dynCall_vi']($0, $1);
  //       }, 0);
  //     }, callback.fn, callback.user_data);

  // https://stackoverflow.com/a/29319440

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unimplemented loop command");

  //   iree_loop_emscripten_scope_t* scope =
  //       (iree_loop_emscripten_scope_t*)self;
  //   iree_loop_emscripten_t* loop_emscripten = scope->loop_emscripten;

  // if (IREE_UNLIKELY(!loop_emscripten->run_ring)) {
  //   return iree_make_status(
  //       IREE_STATUS_FAILED_PRECONDITION,
  //       "new work cannot be enqueued while the loop is shutting down");
  // }

  // TODO(scotttodd): return immediately to make these all tail calls.
  // switch (command) {
  //   case IREE_LOOP_COMMAND_CALL:
  //     return iree_loop_run_ring_enqueue(
  //         loop_emscripten->run_ring,
  //         (iree_loop_run_op_t){
  //             .command = command,
  //             .scope = scope,
  //             .params =
  //                 {
  //                     .call = *(const iree_loop_call_params_t*)params,
  //                 },
  //         });
  //   case IREE_LOOP_COMMAND_DISPATCH:
  //     return iree_loop_run_ring_enqueue(
  //         loop_emscripten->run_ring,
  //         (iree_loop_run_op_t){
  //             .command = command,
  //             .scope = scope,
  //             .params =
  //                 {
  //                     .dispatch = *(const
  //                     iree_loop_dispatch_params_t*)params,
  //                 },
  //         });
  //   case IREE_LOOP_COMMAND_WAIT_UNTIL:
  //     return iree_loop_wait_list_insert(
  //         loop_emscripten->wait_list,
  //         (iree_loop_wait_op_t){
  //             .command = command,
  //             .scope = scope,
  //             .params =
  //                 {
  //                     .wait_until =
  //                         *(const iree_loop_wait_until_params_t*)params,
  //                 },
  //         });
  //   case IREE_LOOP_COMMAND_WAIT_ONE:
  //     return iree_loop_wait_list_insert(
  //         loop_emscripten->wait_list,
  //         (iree_loop_wait_op_t){
  //             .command = command,
  //             .scope = scope,
  //             .params =
  //                 {
  //                     .wait_one = *(const
  //                     iree_loop_wait_one_params_t*)params,
  //                 },
  //         });
  //   case IREE_LOOP_COMMAND_WAIT_ALL:
  //   case IREE_LOOP_COMMAND_WAIT_ANY:
  //     return iree_loop_wait_list_insert(
  //         loop_emscripten->wait_list,
  //         (iree_loop_wait_op_t){
  //             .command = command,
  //             .scope = scope,
  //             .params =
  //                 {
  //                     .wait_multi =
  //                         *(const iree_loop_wait_multi_params_t*)params,
  //                 },
  //         });
  //   case IREE_LOOP_COMMAND_DRAIN:
  //     return iree_loop_emscripten_drain_scope(
  //         loop_emscripten, scope,
  //         ((const iree_loop_drain_params_t*)params)->deadline_ns);
  //   default:
  //     return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
  //                             "unimplemented loop command");
  // }
}
