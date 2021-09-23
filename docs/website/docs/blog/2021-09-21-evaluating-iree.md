---
title: Evaluating IREE
---

 Tuesday, September 21, 2021<br>
 By Scott Todd

```
Planning notes:

* Title brainstorm
  * Evaluating IREE
  * Exploring IREE
  * Exploring compilation
  * Exploring performance
  * Compiler settings and performance
  * Tuning, exploring, evaluating
* Upload all files used in the blog post to a .zip on Drive or GCS
* If posting a Colab notebook, pin the pip install version
* If using native tools, pin the git commit
* Mention that specific tools and how to use them may change, so read for the
  general workflow
* use collapsible regions for details

possible improvements / weird things spotted

* `--iree-llvm-list-targets` doesn't work like `--help` (requires running translation)
  it also exits anyways... so requires setting a throw-away arg
* would like a `--iree-llvm-print-target` to echo the selection / default
* `--iree-llvm-keep-linker-artifacts` -> Dependencies on .dll -> shows floorf and fmaf (https://github.com/google/iree/issues/4717)
* python tracing
  * docs are needed (https://github.com/google/iree/issues/6688)
  * ran into errors enabling in Colab: `%env IREE_SAVE_CALLS=$ARTIFACTS_DIR` -> `TypeError: dump_all() got an unexpected keyword argument 'sort_keys'` -> had to update `!pip3 install --upgrade pyyaml`
  * wanted to record trace in python and re-target against a different vmfb (hack in Colab -> profile in native environment)
  * `iree-run-module` and `iree-run-trace` seem redundant - could they be unified? (want `--print_statistics` from the trace program)
  * put absolute path to compiled .vmfb module file, ran and got no output on console
* `iree-translate` on its own doesn't work in Colab while `iree-import-tflite` does:

  !python -m pip install iree-compiler-snapshot iree-runtime-snapshot iree-tools-tflite-snapshot -f https://github.com/google/iree/releases/latest

  !iree-translate --help
  Traceback (most recent call last):
  File "/usr/local/bin/iree-translate", line 5, in <module>
    from iree.tools.core.scripts.iree_translate.__main__ import main
  ModuleNotFoundError: No module named 'iree.tools.core.scripts'

  (verified) fix in https://github.com/google/iree/pull/7153
```


# Evaluating IREE

TODO: write this

* introduction
* evaluating IREE for [program, compilation target, deployment scenario, device]
* compare performance and execution characteristics with other frameworks
  * binary size for compiled programs
  * execution latency
  * runtime memory usage

## Importing a program from TensorFlow Lite

TODO: write this

* text_classification from Colab notebook? (ready to go integration into an Android app)
* model from our benchmarks?
* model with existing benchmarks from another framework to compare against

## Compiling using `iree-translate`

TODO: write this

* dylib for Android ARM -> my Samsung Galaxy S10?
* dylib for Windows AMD?
* flags
  * `--iree-llvm-list-targets`
  * `--iree-llvm-keep-linker-artifacts`
  * `--iree-llvm-debug-symbols`
  * `--iree-vm-bytecode-module-strip-source-map`
  * `--iree-vm-bytecode-source-listing`
* list binary size for the compiled artifact
* open the compiled .vmfb with 7zip
* poke at the compiled .vmfb with `iree-dump-module`
* `IR Dump Before mlir::iree_compiler::IREE::HAL::SerializeTargetExecutablesPass`

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb

λ zipinfo D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
Archive:  D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
Zip file size: 790062 bytes, number of entries: 1
?---------  3.0 unx   142442 bx stor 80-000-00 00:00 _text_classification_linked_llvm_system_dll_x86_64_binary.fb
1 file, 142442 bytes uncompressed, 142442 bytes compressed:  0.0%

λ iree\tools\iree-dump-module.exe D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb > D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_dump.json
```


With no debug symbols or source maps

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebug.vmfb --iree-llvm-debug-symbols=false --iree-vm-bytecode-module-strip-source-map

λ zipinfo D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebug.vmfb
Archive:  D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebug.vmfb
Zip file size: 661284 bytes, number of entries: 1
?---------  3.0 unx    15398 bx stor 80-000-00 00:00 _text_classification_linked_llvm_system_dll_x86_64_binary.fb
1 file, 15398 bytes uncompressed, 15398 bytes compressed:  0.0%
```

With embedded ELF

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_embeddednodebugsymbols.vmfb --iree-llvm-debug-symbols=false --iree-llvm-link-embedded=true

λ zipinfo D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_embeddednodebugsymbols.vmfb
Archive:  D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_embeddednodebugsymbols.vmfb
Zip file size: 662352 bytes, number of entries: 1
?---------  3.0 unx    14720 bx stor 80-000-00 00:00 _text_classification_linked_llvm_embedded_elf_x86_64_binary.so
1 file, 14720 bytes uncompressed, 14720 bytes compressed:  0.0%
```

With source listing

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_embeddednodebugsymbols.vmfb --iree-llvm-link-embedded=true --iree-vm-bytecode-source-listing=D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_sourcelisting.mlir
```

## Running the compiled program

### Running `iree-run-trace`

pull calls.yml from Colab then edit module.vmfb path to absolute path to local `text_classification_llvmaot.vmfb`

```
λ .\iree\tools\iree-run-trace.exe D:\dev\projects\iree-tmp\blog\calls_modified.yml
--- CALL[module.main] ---
```

### Running `iree-run-module`

TODO: write this

```
λ .\iree\tools\iree-run-module.exe --driver=dylib --entry_function=main --function_input=256xi32 --module_file=D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
I D:\dev\projects\iree\iree\tools\utils\vm_util.cc:185] Creating driver and device for 'dylib'...
EXEC @main
result[0]: hal.buffer_view
1x2xf32=[0.348199 0.651801]
```

```
λ .\iree\tools\iree-run-module.exe --driver=dylib --entry_function=main --function_input=256xi32 --m
odule_file=D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb --print_statistics
I D:\dev\projects\iree\iree\tools\utils\vm_util.cc:185] Creating driver and device for 'dylib'...
EXEC @main
result[0]: hal.buffer_view
1x2xf32=[0.348199 0.651801]
[[ iree_hal_allocator_t memory statistics ]]
  HOST_LOCAL:         1024B peak /         1024B allocated /         1024B freed /            0B live
DEVICE_LOCAL:       657672B peak /       657672B allocated /       657672B freed /            0B live
```

```
.\iree\tools\iree-run-module.exe --driver=dylib --entry_function=main --function_input=256xi32="[[1 13 8 3 117 19 206 109 10 1134 152 2301 385 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]" --module_file=D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
I D:\dev\projects\iree\iree\tools\utils\vm_util.cc:185] Creating driver and device for 'dylib'...
EXEC @main
result[0]: hal.buffer_view
1x2xf32=[0.100271 0.899729]
```

## Profiling runtime performance with Tracy

TODO: write this

* Tracy screenshots
* label regions
* go into statistics and look up which dispatches are taking the most time
* map dispatches back to linalg or tosa ops?

```
set TRACY_NO_EXIT=1
```

then run `iree-run-module` as above

<!--  -->
<!--  -->
---
<!--  -->
<!--  -->

## Appendix

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb --iree-llvm-list-targets
  Registered Targets:
    aarch64    - AArch64 (little endian)
    aarch64_32 - AArch64 (little endian ILP32)
    aarch64_be - AArch64 (big endian)
    arm        - ARM
    arm64      - ARM64 (little endian)
    arm64_32   - ARM64 (little endian ILP32)
    armeb      - ARM (big endian)
    riscv32    - 32-bit RISC-V
    riscv64    - 64-bit RISC-V
    thumb      - Thumb
    thumbeb    - Thumb (big endian)
    wasm32     - WebAssembly 32-bit
    wasm64     - WebAssembly 64-bit
    x86        - 32-bit X86: Pentium-Pro and above
    x86-64     - 64-bit X86: EM64T and AMD64
```

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classifica
tion.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-t
ype=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb --iree-llvm-keep-linker-
artifacts
D:\dev\projects\iree-data\models\text_classification\text_classification.mlir:1:1: remark: linker ar
tifacts for system_dll_x86_64 preserved:
    C:\Users\Scott\AppData\Local\Temp\text_classification_linked_llvm-da70d5.dll
module  {
^
```
