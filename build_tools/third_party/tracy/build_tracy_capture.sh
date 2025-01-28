#!/bin/bash

THIS_DIR="$(cd $(dirname $0) && pwd)"
cd "${THIS_DIR}/../../../third_party/tracy"

cmake -B capture/build -S capture -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build capture/build --config Release --parallel
