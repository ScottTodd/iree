# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

#-------------------------------------------------------------------------------
# Sync
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_web_sample_dynamic_sync")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    device_sync.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-dynamic-sync")

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  iree_runtime_runtime
  iree_hal_local_loaders_system_library_loader
  iree_hal_local_sync_driver
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_load_program', '_inspect_program', '_unload_program', '_call_function']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # Programs loaded dynamically can require additional memory, so allow growth.
  "-sALLOW_MEMORY_GROWTH"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
  #
  # Dynamic linking: https://emscripten.org/docs/compiling/Dynamic-Linking.html
  "-sMAIN_MODULE=2"
  # "-sALLOW_TABLE_GROWTH"
)

#-------------------------------------------------------------------------------
# Multithreaded
#-------------------------------------------------------------------------------

set(_NAME "iree_experimental_web_sample_dynamic_multithreaded")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    device_multithreaded.c
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "web-sample-dynamic-multithreaded")

# Note: we have to be very careful about dependencies here.
#
# The general purpose libraries link in multiple executable loaders and HAL
# drivers/devices, which include code not compatible with Emscripten.
target_link_libraries(${_NAME}
  iree_runtime_runtime
  iree_hal_local_executable_library
  iree_hal_local_loaders_system_library_loader
  iree_hal_local_task_driver
  iree_task_api
)

target_link_options(${_NAME} PRIVATE
  # https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-ccall-cwrap
  # "-sEXPORTED_FUNCTIONS=['_setup_sample', '_cleanup_sample', '_load_program', '_inspect_program', '_unload_program', '_call_function']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  #
  "-sASSERTIONS=1"
  #
  # https://developer.chrome.com/blog/wasm-debugging-2020/
  "-g"
  "-gseparate-dwarf"
  #
  # Dynamic linking: https://emscripten.org/docs/compiling/Dynamic-Linking.html
  "-sMAIN_MODULE"
  # "-sALLOW_TABLE_GROWTH"
  #
  # error: undefined symbol: __call_main (referenced by top-level compiled C/C++ code)
  # warning: Link with `-sLLD_REPORT_UNDEFINED` to get more information on undefined symbols
  # warning: To disable errors for undefined symbols use `-sERROR_ON_UNDEFINED_SYMBOLS=0`
  # warning: ___call_main may need to be added to EXPORTED_FUNCTIONS if it arrives from a system libraryerror: undefined symbol: cpuinfo_emscripten_init (referenced by top-level compiled C/C++ code)
  # warning: _cpuinfo_emscripten_init may need to be added to EXPORTED_FUNCTIONS if it arrives from a system library
  "-sERROR_ON_UNDEFINED_SYMBOLS=0"
  #
  # ------------------------------------------------------------------------- #
  # Multithreading with pthreads, built on Web Workers and SharedArrayBuffer.
  # Docs: https://emscripten.org/docs/porting/pthreads.html
  # IREE's use of threading is pretty restricted so we could eventually drop
  # Emscripten's pthreads implementation in favor of our own.
  #
  # Note: this -pthread flag also needs to be set in compile options.
  "-pthread"
  #
  # IREE creates long lived worker threads during device creation around system
  # startup time. No fixed size thread pool is necessary, so disable thread
  # pool size checks and do not specify the pool size.
  # By starting IREE from a Web Worker, we can avoid needing `PROXY_TO_PTHREAD`
  # and can block when interacting with threads.
  # Web Worker creation requires yielding back to the main browser event loop,
  # so splitting program execution across multiple JavaScript function calls
  # (e.g. `createDevice(); runInference();`) also helps avoid deadlocks.
  "-sPTHREAD_POOL_SIZE_STRICT=0"
  # "-sPTHREAD_POOL_SIZE=N"
  # "-sPROXY_TO_PTHREAD"
  #
  # Threads use memory, so using a larger pool of memory or allowing memory
  # growth may be necessary when using `emscripten_num_logical_cores()`
  # or `navigator.hardwareConcurrency` to pick the worker thread count.
  # IREE is pretty good about not allocating outside of startup, so concerns
  # about this causing slow access to memory *may* not affect IREE too much.
  "-sALLOW_MEMORY_GROWTH=1"
  # ------------------------------------------------------------------------- #
)

# Need to use Emscripten's `SIDE_MODULE` option to generate libraries that can
# be dynamically linked in while using pthreads
#
# TypeError: tlsInitFunc is not a function
# https://github.com/emscripten-core/emscripten/blob/0c4fba0371abc8c7219dc481a02ea4a464b14641/src/library_dylink.js#L574-L576
# https://github.com/emscripten-core/emscripten/blob/main/system/lib/pthread/emscripten_tls_init.c
