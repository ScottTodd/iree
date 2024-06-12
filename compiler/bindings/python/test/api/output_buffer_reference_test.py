# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This test specifically exercises the VmModule.wrap_buffer method in combination with
# the memory mapped buffers produced by the in-process compile API.
# The ownership transfers can be tricky and this exercises some of the corners.

import gc

import iree.compiler
from iree.compiler.api import (
    Session,
    Source,
    Output,
)

from iree.runtime import VmContext, VmInstance, VmModule, load_vm_module


def compile_simple_mul_binary() -> Output:
    asm = b"""
    module @arithmetic {
    func.func @simple_mul(%arg0: f32, %arg1: f32) -> f32 {
        %0 = arith.mulf %arg0, %arg1 : f32
        return %0 : f32
    }
    }
    """

    output = Output.open_membuffer()
    session = Session()
    source = Source.wrap_buffer(session, asm)
    session.set_flags(
        f"--iree-hal-target-backends={iree.compiler.core.DEFAULT_TESTING_BACKENDS[0]}"
    )
    inv = session.invocation()
    inv.parse_source(source)
    inv.execute()
    inv.output_vm_bytecode(output)
    return output


vmfb_contents = bytes(compile_simple_mul_binary().map_memory())
# This print is a load bearing part of the test. It ensures that the memory
# is readable.
print("VMFB Length =", len(vmfb_contents), vmfb_contents)


def run_mmap_free_before_context_test():
    instance = VmInstance()
    output = Output.open_membuffer()
    output.write(vmfb_contents)
    mapped_memory = output.map_memory()
    module = VmModule.wrap_buffer(instance, mapped_memory)
    # context = VmContext(instance, modules=[module])
    loaded_module = load_vm_module(module)
    # Shutdown in the most egregious way possible.
    # Note that during context destruction, the context needs some final
    # access to the mapped memory to run destructors. It is easy for the
    # reference to the backing memory to be invalid at this point, thus
    # this test.
    print("gc.collect() then `output = None`")
    gc.collect()
    output = None
    print("gc.collect() then `mapped_memory = None`")
    gc.collect()
    mapped_memory = None
    print("gc.collect() then `module = None`")
    gc.collect()
    module = None
    print("gc.collect() then `context = None`")
    gc.collect()
    context = None
    gc.collect()
    print("run_mmap_free_before_context_test end")


# for i in range(10):
#     run_mmap_free_before_context_test()
run_mmap_free_before_context_test()
