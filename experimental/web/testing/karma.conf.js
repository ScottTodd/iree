// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Documentation:
//   * http://karma-runner.github.io/6.4/intro/configuration.html
//   * http://karma-runner.github.io/6.4/config/configuration-file.html
//   * https://www.npmjs.com/package/karma-chrome-launcher
//   * https://github.com/puppeteer/puppeteer
//
// Other references:
//   * https://github.com/BabylonJS/Babylon.js/blob/master/jest-puppeteer.config.js
//   * https://github.com/Kitware/vtk-js/blob/master/karma.conf.js

// TODO(scotttodd): Use puppeteer to get a bundled version of Chromium
//                  (first testing with a local, manually installed browser)
//   process.env.CHROME_BIN = require('puppeteer').executablePath()

module.exports = function(config) {
  config.set({
    browsers: ['ChromeCanary', 'ChromeCanary_with_WebGPU'],

    customLaunchers: {
      ChromeCanary_with_WebGPU:
          {base: 'ChromeCanary', flags: ['--enable-unsafe-webgpu']}
    },

    // TODO(scotttodd): file patterns (relative to directory?)
    files: [
      {
        pattern:
            'D:/dev/projects/iree/build-emscripten/runtime/src/iree/base/bitfield_test.js'
      },
      {
        pattern:
            'D:/dev/projects/iree/build-emscripten/runtime/src/iree/base/bitfield_test.wasm',
        included: false
      },
      // { pattern: 'test/testdata-file-cache-*.json', included: false },
      // { pattern: 'test/data/**/*', included: false, nocache: true },
      // { pattern: 'dist/ort-wasm.wasm', included: false },
      // { pattern: 'dist/ort-wasm-threaded.wasm', included: false },
      // { pattern: 'dist/ort-wasm-simd.wasm', included: false },
      // { pattern: 'dist/ort-wasm-simd-threaded.wasm', included: false },
      // { pattern: 'dist/ort-wasm-threaded.worker.js', included: false },
    ],
  })
}
