// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var LibraryLoopEmscripten = {
  loop_emscripten_log_2: function() {
    console.log('hello from loop_emscripten.js!');
  },
}

// autoAddDeps(LibraryLoopEmscripten, '$LoopEmscripten');
mergeInto(LibraryManager.library, LibraryLoopEmscripten);
