#!/bin/bash

THIS_DIR="$(cd $(dirname $0) && pwd)"
cd "${THIS_DIR}/../../../third_party/tracy"

cmake -B profiler/build -S profiler -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build profiler/build --config Release --parallel
