 Tuesday, September 21, 2021<br>
 By Scott Todd

```
Planning notes:

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
D:\dev\projects\iree-build
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb

λ zipinfo D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
Archive:  D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb
Zip file size: 790062 bytes, number of entries: 1
?---------  3.0 unx   142442 bx stor 80-000-00 00:00 _text_classification_linked_llvm_system_dll_x86_64_binary.fb
1 file, 142442 bytes uncompressed, 142442 bytes compressed:  0.0%

λ iree\tools\iree-dump-module.exe D:\dev\projects\iree-tmp\blog\text_classification_llvmaot.vmfb > D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_dump.json
```


With no debug symbols

```
λ iree\tools\iree-translate.exe D:\dev\projects\iree-data\models\text_classification\text_classification.mlir --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-input-type=tosa --o D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebugsymbols.vmfb --iree-llvm-debug-symbols=false

λ zipinfo D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebugsymbols.vmfb
Archive:  D:\dev\projects\iree-tmp\blog\text_classification_llvmaot_nodebugsymbols.vmfb
Zip file size: 663022 bytes, number of entries: 1
?---------  3.0 unx    15398 bx stor 80-000-00 00:00 _text_classification_linked_llvm_system_dll_x86_64_binary.fb
1 file, 15398 bytes uncompressed, 15398 bytes compressed:  0.0%
```

## Running the compiled program with `iree-run-module` and `iree-run-trace`

TODO: write this

* `--print_statistics` with `iree-run-module`

## Profiling runtime performance with Tracy

TODO: write this

* Tracy screenshots
* label regions
* go into statistics and look up which dispatches are taking the most time
* map dispatches back to linalg or tosa ops?


<!--  -->
<!--  -->
---
<!--  -->
<!--  -->

## Appendix

```
D:\dev\projects\iree-build
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
