// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://jasmine.github.io/tutorials/async

describe('A suite is just a function', function() {
  let a;

  it('and so is a spec', function() {
    a = true;
    console.log('and so is a spec');

    expect(a).toBe(true);
  });

  it('does something async', async function() {
    return promiseReturningFunction().then((result) => {
      expect(result).toEqual(5);
    })
  });

  function promiseReturningFunction() {
    return new Promise((resolve, reject) => {
      resolve(5);
    });
  }
});
