#!/bin/bash

set -eu -o errtrace

THIS_DIR="$(cd $(dirname $0) && pwd)"
cd "${THIS_DIR}/../../../third_party/tracy"

echo "Deleting old build directories from '$(pwd)'"
set -x
rm -rf profiler/build
rm -rf capture/build
rm -rf csvexport/build
