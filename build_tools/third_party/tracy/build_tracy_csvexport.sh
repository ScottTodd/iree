#!/bin/bash

THIS_DIR="$(cd $(dirname $0) && pwd)"
cd "${THIS_DIR}/../../../third_party/tracy"

cmake -B csvexport/build -S csvexport -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build csvexport/build --config Release --parallel
