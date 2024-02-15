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
    # /test_data_set_0/input_2.pb
    # /test_data_set_0/output_0.pb  (usually only one)
    # /test_data_set_0/output_1.pb
    # /test_data_set_0/output_2.pb

    # root_dir_path.
    pass


if __name__ == "__main__":
    test_dirs = find_tests()

    run_test(test_dirs[0])
