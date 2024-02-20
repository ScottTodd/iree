# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import onnx
from pathlib import Path
from onnx import numpy_helper
import subprocess
import numpy as np
import sys


GENERATED_TESTS_ROOT_DIR = "D:/dev/projects/onnx/onnx/backend/test/data/node"
OUTPUT_ROOT_DIR = "D:/dev/projects/iree-tmp/2024_02_19_onnx"


def find_tests():
    # for root, dirs, files in os.walk(GENERATED_TESTS_ROOT_DIR):
    #     print(root, dirs, files)

    root_dir_path = Path(GENERATED_TESTS_ROOT_DIR)
    test_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
    print(f"Found {len(test_dir_paths)} tests")

    # for dir in root_dir_path.iterdir():
    #     if not dir.is_dir():
    #         continue
    #     print("Discovered directory:", dir.name)

    return test_dir_paths


def convert_onnx_files(test_dir_path, converted_dir_path):
    # This converts one 'test_[name]' subfolder from this:
    #
    #   onnx/backend/test/data/node/...  (GENERATED_TESTS_ROOT_DIR)
    #     test_[name]/
    #       model.onnx
    #       test_data_set_0/
    #         input_0.pb
    #         output_0.pb
    #
    # to this:
    #
    #   converted_dir_path/...
    #     test_[name]/
    #       model.mlir  (torch-mlir)
    #       input_0.npy
    #       output_0.npy
    #       test_data_flags.txt  (flagfile with --input=, --expected_output=)

    converted_dir_path.mkdir(parents=True, exist_ok=True)
    # print(
    #     f"Converting test files\n  from '{test_dir_path}'\n  to '{converted_dir_path}'"
    # )

    test_data_flagfile_path = converted_dir_path / "test_data_flags.txt"
    test_data_flagfile_lines = []

    # Import model.onnx to model.mlir.
    onnx_model_path = test_dir_path / "model.onnx"
    converted_model_path = converted_dir_path / "model.mlir"
    exec_args = [
        "iree-import-onnx",
        str(onnx_model_path),
        "-o",
        str(converted_model_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        print(f"  {converted_dir_path.name[5:]} import failed", file=sys.stderr)
        return False

    test_data_dirs = sorted(test_dir_path.glob("test_data_set*"))
    if len(test_data_dirs) != 1:
        print("WARNING: unhandled 'len(test_data_dirs) != 1'")
        return False

    # Convert input_*.pb and output_*.pb to .npy files.
    test_data_dir = test_data_dirs[0]
    test_inputs = list(test_data_dir.glob("input_*.pb"))
    test_outputs = list(test_data_dir.glob("output_*.pb"))
    model = onnx.load(onnx_model_path)
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        t = convert_io_proto(test_input, model.graph.input[i].type)
        if t is None:
            return False
        input_path = (converted_dir_path / test_input.stem).with_suffix(".npy")
        np.save(input_path, t, allow_pickle=False)
        test_data_flagfile_lines.append(f"--input=@{input_path.name}\n")
    for i in range(len(test_outputs)):
        test_output = test_outputs[i]
        t = convert_io_proto(test_output, model.graph.output[i].type)
        if t is None:
            return False
        output_path = (converted_dir_path / test_output.stem).with_suffix(".npy")
        np.save(output_path, t, allow_pickle=False)
        test_data_flagfile_lines.append(f"--expected_output=@{output_path.name}\n")

    with open(test_data_flagfile_path, "wt") as f:
        f.writelines(test_data_flagfile_lines)

    return True


def convert_io_proto(proto_filename, type_proto):

    with open(proto_filename, "rb") as f:
        protobuf_content = f.read()
        if type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            t = numpy_helper.to_array(tensor)
            return t
        else:
            print(f"Unsupported proto type: {type_proto}")
            return None


def compile_onnx_tests(converted_dir_path):
    # print(f"Compiling converted tests for dir: '{converted_dir_path.name}'")

    converted_model_path = converted_dir_path / "model.mlir"
    compiled_model_path = converted_dir_path / "model_cpu.vmfb"
    exec_args = [
        "iree-compile",
        str(converted_model_path),
        "--iree-hal-target-backends=llvm-cpu",
        "-o",
        str(compiled_model_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        # print(f"  Compile failed,\n    stdout: {ret.stdout},\n    stderr: {ret.stderr}")
        print(f"  {converted_dir_path.name[5:]} compile failed", file=sys.stderr)
        return False

    config_flagfile_path = converted_dir_path / "config_cpu_flags.txt"
    config_flagfile_lines = []
    config_flagfile_lines.append("--device=local-task\n")
    config_flagfile_lines.append(f"--module={compiled_model_path.name}\n")
    with open(config_flagfile_path, "wt") as f:
        f.writelines(config_flagfile_lines)

    return True


def run_onnx_tests(converted_dir_path):
    # print(f"Running compiled tests for dir: '{converted_dir_path.name}'")

    config_flagfile_path = converted_dir_path / "config_cpu_flags.txt"
    test_data_flagfile_path = converted_dir_path / "test_data_flags.txt"

    exec_args = [
        "iree-run-module",
        f"--flagfile={config_flagfile_path.name}",
        f"--flagfile={test_data_flagfile_path.name}",
    ]
    # print("  Exec:", " ".join(exec_args))
    ret = subprocess.run(exec_args, capture_output=True, cwd=converted_dir_path)
    if ret.returncode != 0:
        print(f"  {converted_dir_path.name[5:]} run failed", file=sys.stderr)
        return False

    return True


def run_test(test_dir_path):
    # print(f"-----------------------------------------------------------------")
    # print(f"Running test for dir: '{test_dir_path.name}'")

    converted_dir_path = Path(OUTPUT_ROOT_DIR) / test_dir_path.name
    convert_result = convert_onnx_files(test_dir_path, converted_dir_path)
    if not convert_result:
        return False
    compile_result = compile_onnx_tests(converted_dir_path)
    if not compile_result:
        return False
    run_result = run_onnx_tests(converted_dir_path)
    if not run_result:
        return False

    return True


if __name__ == "__main__":
    test_dir_paths = find_tests()

    print("******************************************************************")
    pass_count = 0
    fail_count = 0
    # Toggle comment to limit how many tests run
    # for i in range(10):
    for i in range(len(test_dir_paths)):
        current_number = str(i).rjust(4, "0")
        number_str = f"[{current_number}/{len(test_dir_paths)}]"

        result = run_test(test_dir_paths[i])
        if result:
            print(f"{number_str}: {test_dir_paths[i].name} PASS")
            pass_count += 1
        else:
            print(f"{number_str}: {test_dir_paths[i].name} FAIL")
            fail_count += 1
    print("******************************************************************")
    print(f"Pass count: {pass_count}")
    print(f"Fail count: {fail_count}")
