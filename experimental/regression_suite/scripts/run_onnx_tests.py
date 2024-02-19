# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import onnx
from pathlib import Path
from onnx import numpy_helper
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
    test_dirs = [p for p in root_dir_path.iterdir() if p.is_dir()]
    print("test_dirs.length:", len(test_dirs))

    # for dir in root_dir_path.iterdir():
    #     if not dir.is_dir():
    #         continue
    #     print("Discovered directory:", dir.name)

    return test_dirs


def convert_io_proto(proto_filename, type_proto):
    output_dir_path = Path(OUTPUT_ROOT_DIR)
    output_file_path = output_dir_path / proto_filename.stem
    print(f"Converting proto file '{proto_filename}' to '{output_file_path}'")

    with open(proto_filename, "rb") as f:
        protobuf_content = f.read()

        if type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            t = numpy_helper.to_array(tensor)
            # assert isinstance(t, np.ndarray)
            # target_list.append(t)
            print("  tensor_type -> TensorProto")
            # print(f"   tensor: {t}")

            np.save(output_file_path, t, allow_pickle=False)
            print(f"    Saved to {output_file_path}")
        else:
            print(f"Unsupported proto type: {type_proto}")


def run_test(test_dir):
    print(f"Running test for dir: '{test_dir.name}'")

    # /model.onnx
    # /test_data_set_0/
    # /test_data_set_0/input_0.pb  (usually only one)
    # /test_data_set_0/input_1.pb

    model_path = test_dir / "model.onnx"

    # TODO(scotttodd): glob for test_data_set*
    test_data_dir = test_dir / "test_data_set_0"
    inputs = list(test_data_dir.glob("input_*.pb"))
    outputs = list(test_data_dir.glob("output_*.pb"))

    print(
        f"Testing model {model_path.name} with {inputs[0].name} and {outputs[0].name}"
    )

    model = onnx.load(model_path)
    convert_io_proto(inputs[0], model.graph.input[0].type)
    convert_io_proto(outputs[0], model.graph.output[0].type)

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
    test_dirs = find_tests()

    run_test(test_dirs[0])
