// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the JavaScript side of loop_emscripten.c
//
// References:
//   * https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html
//   * https://github.com/evanw/emscripten-library-generator
//   * https://github.com/emscripten-core/emscripten/tree/main/src

const LibraryLoopEmscripten = {
  $loop_emscripten_support__postset: 'loop_emscripten_support();',
  $loop_emscripten_support: function() {
    const IREE_STATUS_OK = 0;
    const IREE_STATUS_CODE_MASK = 0x1F;
    const IREE_STATUS_ABORTED = 10 & IREE_STATUS_CODE_MASK;
    const IREE_STATUS_OUT_OF_RANGE = 11 & IREE_STATUS_CODE_MASK;

    class LoopEmscriptenScope {
      constructor() {
        this.isAlive = true;
        this.nextOperationId = 0;

        // Dictionary of operationIds -> timeoutIds.
        this.pendingOperations = {};
      }

      destroy() {
        this.isAlive = false;

        for (const [_, timeoutId] of Object.entries(this.pendingOperations)) {
          clearTimeout(timeoutId);
        }
        this.pendingOperations = {};
      }

      command_call(callback, user_data, loop) {
        const operationId = this.nextOperationId++;
        console.log('starting operation id:', operationId);

        const timeoutId = setTimeout(() => {
          console.log(
              'finished operation id:', operationId,
              ', isAlive:', this.isAlive);

          // TODO(scotttodd): sticky failure with IREE_STATUS_ABORTED?
          if (!this.isAlive) return;

          const ret =
              Module['dynCall_iiii'](callback, user_data, loop, IREE_STATUS_OK);
          // TODO(scotttodd): handle the returned status (sticky failure state?)
        }, 0);
        this.pendingOperations[operationId] = timeoutId;

        return IREE_STATUS_OK;
      }
    }

    class LoopEmscripten {
      constructor() {
        this.nextScopeHandle = 0;

        // Dictionary of scopeHandles -> LoopEmscriptenScopes.
        this.scopes = {};
      }

      loop_allocate_scope() {
        const scopeHandle = this.nextScopeHandle++;
        this.scopes[scopeHandle] = new LoopEmscriptenScope();
        console.log('Created LoopEmscriptenScope, handle:', scopeHandle);
        console.log('Scopes:', this.scopes);
        return scopeHandle;
      }

      loop_free_scope(scope_handle) {
        if (!(scope_handle in this.scopes)) return;

        console.log('loop_free_scope with scope handle:', scope_handle);

        const scope = this.scopes[scope_handle];
        scope.isAlive = false;
        // TODO(scotttodd): assert empty?

        delete this.scopes[scope_handle];
      }

      loop_command_call(scope_handle, callback, user_data, loop) {
        console.log('loop_command_call with scope handle:', scope_handle);
        if (!(scope_handle in this.scopes)) return IREE_STATUS_OUT_OF_RANGE;
        const scope = this.scopes[scope_handle];
        return scope.command_call(callback, user_data, loop);
      }
    }

    const instance = new LoopEmscripten();
    _loop_allocate_scope = instance.loop_allocate_scope.bind(instance);
    _loop_free_scope = instance.loop_free_scope.bind(instance);
    _loop_command_call = instance.loop_command_call.bind(instance);
  },

  loop_allocate_scope: function() {},
  loop_allocate_scope__deps: ['$loop_emscripten_support'],
  loop_free_scope: function() {},
  loop_free_scope__deps: ['$loop_emscripten_support'],
  loop_command_call: function() {},
  loop_command_call__deps: ['$loop_emscripten_support'],
}

mergeInto(LibraryManager.library, LibraryLoopEmscripten);
