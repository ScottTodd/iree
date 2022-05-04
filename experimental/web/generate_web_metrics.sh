#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generate metrics for a set of programs using IREE's WebAssembly backend.
#
# Intended for interactive developer use, so this generates lots of output
# files for manual inspection outside of the script.
#
# Script steps:
#   * Install IREE tools into a Python virtual environment (venv)
#   * Download program source files (.tflite files from GCS)
#   * Import programs into MLIR (.tflite -> .mlir)
#   * Compile programs for WebAssembly (.mlir -> .vmfb, intermediates)
#   * (TODO) Print statistics that can be produced by a script (i.e. no
#     launching a webpage and waiting for benchmark results there)
#
# Sample usage:
#   generate_web_metrics.sh /tmp/iree/web_metrics/
#
#   then look in that directory for any files you want to process further

set -e

TARGET_DIR="$1"
if [ -z "$TARGET_DIR" ]; then
  echo "ERROR: Expected target directory (e.g. /tmp/iree/web_metrics/)"
  exit 1
fi
mkdir -p ${TARGET_DIR}
cd ${TARGET_DIR}

# Print all commands and the UTC time.
# set -xeuo pipefail
# export PS4='[$(date -u "+%T %Z")] '

###############################################################################
# Set up Python virtual environment                                           #
###############################################################################

python -m venv .venv
source .venv/bin/activate
trap "deactivate 2> /dev/null" EXIT

# Install packages when you want by uncommenting this. Freezing to a specific
# version when iterating on metrics is useful, and fetching is slow anyways.

# python -m pip install --upgrade \
#   --find-links https://github.com/google/iree/releases \
#   iree-compiler iree-tools-tflite iree-tools-xla

###############################################################################
# Download program source files                                               #
###############################################################################

wget -nc https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite
wget -nc https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite

###############################################################################
# Import programs into MLIR                                                   #
###############################################################################

IREE_IMPORT_TFLITE_PATH=iree-import-tflite

# import_program helper
#   Args: program_name, tflite_source_path
#   Imports tflite_source_path to program_name.mlir
function import_program {
  OUTPUT_FILE=./$1_tosa.mlir
  echo "  Importing '$1' to '${OUTPUT_FILE}'..."
  ${IREE_IMPORT_TFLITE_PATH?} $2 -o ${OUTPUT_FILE}
}

import_program "mobilebert_squad" "./mobilebert-baseline-tf2-float.tflite"
import_program "mobilenet_v2" "./mobilenet_v2_1.0_224.tflite"

###############################################################################
# Compile programs                                                            #
###############################################################################

# Either build from source (setting this path), or use from the python packages.
IREE_COMPILE_PATH=~/code/iree-build/iree/tools/iree-compile
# IREE_COMPILE_PATH=iree-compile

# compile_program helper
#   Args: program_name
#   Compiles program_name_tosa.mlir to program_name_wasm.vmfb, dumping
#   statistics and intermediate files to disk.
function compile_program {
  INPUT_FILE=./$1_tosa.mlir
  OUTPUT_FILE=./$1_wasm.vmfb
  echo "  Compiling '${INPUT_FILE}' to '${OUTPUT_FILE}'..."

  EXECUTABLES_DIR=./$1-executables/
  STATISTICS_DIR=./$1-statistics/
  mkdir -p ${EXECUTABLES_DIR}
  mkdir -p ${STATISTICS_DIR}

  # Compile from .mlir to .vmfb, dumping all the intermediate files and
  # compile-time statistics that we can.
  ${IREE_COMPILE_PATH?} ${INPUT_FILE} \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm \
    --iree-llvm-target-triple=wasm32-unknown-emscripten \
    --iree-hal-dump-executable-sources-to=${EXECUTABLES_DIR} \
    --iree-hal-dump-executable-binaries-to=${EXECUTABLES_DIR} \
    --iree-scheduling-dump-statistics-format=csv \
    --iree-scheduling-dump-statistics-file=${STATISTICS_DIR}/$1_statistics.csv \
    --o ${OUTPUT_FILE}

  # Compress the .vmfb file (ideally it would be compressed already, but we can
  # expect some compression support from platforms like the web).
  gzip -k -f ${OUTPUT_FILE}
}

compile_program "mobilebert_squad"
compile_program "mobilenet_v2"

###############################################################################
# TODO: collect/summarize statistics (manual inspection or scripted)
#   * .vmfb size
#   * number of executables
#   * size of each executable (data size)
#   * size of constants
