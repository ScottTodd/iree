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


GENERATED_TESTS_ROOT_DIR = "D:/dev/projects/onnx/onnx/backend/test/data/node"
OUTPUT_ROOT_DIR = "D:/dev/projects/iree-tmp/2024_02_19_onnx"

# iree-import-onnx D:\dev\projects\onnx\onnx\backend\test\data\node\test_abs\model.onnx -o D:\dev\projects\iree-tmp\2024_02_19_onnx\model_torch.mlir
# iree-compile D:\dev\projects\iree-tmp\2024_02_19_onnx\model_torch.mlir --iree-hal-target-backends=llvm-cpu -o D:\dev\projects\iree-tmp\2024_02_19_onnx\model_cpu.vmfb
# iree-run-module --module=D:\dev\projects\iree-tmp\2024_02_19_onnx\model_cpu.vmfb --device=local-task --input=@D:\dev\projects\iree-tmp\2024_02_19_onnx\input_0.npy --expected_output=@D:\dev\projects\iree-tmp\2024_02_19_onnx\output_0.npy


def find_tests():
    # for root, dirs, files in os.walk(GENERATED_TESTS_ROOT_DIR):
    #     print(root, dirs, files)

    root_dir_path = Path(GENERATED_TESTS_ROOT_DIR)
    test_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
    print("test_dir_paths.length:", len(test_dir_paths))

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
    print(
        f"Converting test files\n  from '{test_dir_path}'\n  to '{converted_dir_path}'"
    )

    flagfile_path = converted_dir_path / "test_data_flags.txt"
    flagfile_lines = []

    # Import model.onnx to model.mlir.
    model_path = test_dir_path / "model.onnx"
    converted_model_path = converted_dir_path / "model.mlir"
    exec_args = [
        "iree-import-onnx",
        str(model_path),
        "-o",
        str(converted_model_path),
    ]
    subprocess.run(exec_args, check=True, capture_output=True)

    test_data_dirs = sorted(test_dir_path.glob("test_data_set*"))
    if len(test_data_dirs) != 1:
        print("WARNING: unhandled 'len(test_data_dirs) != 1'")
        return

    # Convert input_*.pb and output_*.pb to .npy files.
    test_data_dir = test_data_dirs[0]
    test_inputs = list(test_data_dir.glob("input_*.pb"))
    test_outputs = list(test_data_dir.glob("output_*.pb"))
    model = onnx.load(model_path)
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        t = convert_io_proto(test_input, model.graph.input[i].type)
        if t is None:
            return False
        input_path = (converted_dir_path / test_input.stem).with_suffix(".npy")
        np.save(input_path, t, allow_pickle=False)
        flagfile_lines.append(f"--input=@{input_path.name}\n")
    for i in range(len(test_outputs)):
        test_output = test_outputs[i]
        t = convert_io_proto(test_output, model.graph.output[i].type)
        if t is None:
            return False
        output_path = (converted_dir_path / test_output.stem).with_suffix(".npy")
        np.save(output_path, t, allow_pickle=False)
        flagfile_lines.append(f"--expected_output=@{output_path.name}\n")

    with open(flagfile_path, "wt") as f:
        f.writelines(flagfile_lines)

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


def run_onnx_tests(converted_dir_path):
    print(f"Running converted tests for dir: '{converted_dir_path.name}'")


def run_test(test_dir_path):
    print(f"-----------------------------------------------------------------")
    print(f"Running test for dir: '{test_dir_path.name}'")

    # /model.onnx
    # /test_data_set_0/
    # /test_data_set_0/input_0.pb  (usually only one)
    # /test_data_set_0/input_1.pb

    converted_dir_path = Path(OUTPUT_ROOT_DIR) / test_dir_path.name
    converted = convert_onnx_files(test_dir_path, converted_dir_path)
    if converted:
        run_onnx_tests(converted_dir_path)

    # model_path = test_dir_path / "model.onnx"

    # # TODO(scotttodd): glob for test_data_set*
    # test_data_dir = test_dir_path / "test_data_set_0"
    # inputs = list(test_data_dir.glob("input_*.pb"))
    # outputs = list(test_data_dir.glob("output_*.pb"))

    # print(
    #     f"Testing model {model_path.name} with {inputs[0].name} and {outputs[0].name}"
    # )

    # model = onnx.load(model_path)
    # convert_io_proto(inputs[0], model.graph.input[0].type)
    # convert_io_proto(outputs[0], model.graph.output[0].type)

    # iree-import-onnx model.onnx -o model_torch.mlir
    # iree-compile model_torch.mlir --iree-hal-target-backends=... -o model_*.vmfb
    # input_*.pb and output_*.pb -> input.bin, output.bin

    # for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
    # inputs = []
    # inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
    # for i in range(inputs_num):
    #     input_file = os.path.join(test_data_dir, f"input_{i}.pb")
    #     self._load_proto(input_file, inputs, model.graph.input[i].type)
    # ref_outputs = []
    # ref_outputs_num = len(
    #     glob.glob(os.path.join(test_data_dir, "output_*.pb"))
    # )
    # for i in range(ref_outputs_num):
    #     output_file = os.path.join(test_data_dir, f"output_{i}.pb")
    #     self._load_proto(
    #         output_file, ref_outputs, model.graph.output[i].type
    #     )
    # outputs = list(prepared_model.run(inputs))
    # self.assert_similar_outputs(
    #     ref_outputs,
    #     outputs,
    #     rtol=model_test.rtol,
    #     atol=model_test.atol,
    #     model_dir=model_dir,
    # )
    # /test_data_set_0/input_2.pb
    # /test_data_set_0/output_0.pb  (usually only one)
    # /test_data_set_0/output_1.pb
    # /test_data_set_0/output_2.pb

    # root_dir_path.
    pass


if __name__ == "__main__":
    test_dir_paths = find_tests()

    run_test(test_dir_paths[0])
