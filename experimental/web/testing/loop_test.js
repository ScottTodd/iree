// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://jasmine.github.io/tutorials/async

describe('Loop tests', function() {
  // TODO(scotttodd): use beforeAll instead? Emscripten modules are stateful...
  beforeEach(async function() {
    return new Promise((resolve, reject) => {
      var Module = {
        'onRuntimeInitialized': () => {
          resolve();
        }
      };

      // load a test module with exported methods
      // export all test methods explicitly?
      // export getTestFn(index) and getTestFnsCount()?

      // JS tests that call exported functions on the module then wait on
      // promises or on callbacks

      // note: 'require' is nodejs only
      // require('build-emscripten/runtime/src/iree/base/loop_test.js');
    });
  });

  it('sample test', function() {
    a = true;
    expect(a).toBe(true);
  });
});
