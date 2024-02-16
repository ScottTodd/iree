from pathlib import Path
import os

GENERATED_TESTS_ROOT_DIR = "D:/dev/projects/onnx/onnx/backend/test/data/node"


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

    # TODO(scotttodd): handle > 1 input/output
    input = inputs[0]
    output = outputs[0]

    print(
        f"Testing model {model_path.name} with {inputs[0].name} and {outputs[0].name}"
    )

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
