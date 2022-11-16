// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reference:
// https://github.com/emscripten-core/emscripten/blob/main/src/library_webgpu.js

var LibraryLoopEmscripten = {
  loop_emscripten_log_2: function() {
    console.log('hello from loop_emscripten.js!');
  },

  loopCommandCall: function(callback, userData, loop) {
    console.log('loopCommandCall');

    setTimeout(() => {
      console.log('loopCommandCall -> timeout, calling function');
      const ret =
          Module['dynCall_iiii'](callback, userData, loop, /*status=*/ 0);
      console.log('function result:', ret);
    }, 0);

    return 0;
  },
}

mergeInto(LibraryManager.library, LibraryLoopEmscripten);
