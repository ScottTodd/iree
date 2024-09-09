# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ccache (https://ccache.dev/) setup, with read/write + local/remote options.
#
# Defaults to only reading from the shared remote cache (hosted on GCS) used by
# our Linux CI. The postsubmit CI writes to the cache, for presubmit CI and
# local builds to read from.
#
# Local caches can also be used to interface with external remote caches
# (like https://github.com/actions/cache) by
#   1. downloading the cache directory
#   2. sourcing with IREE_READ_LOCAL_CCACHE=1 IREE_WRITE_LOCAL_CCACHE=[0,1]
#   3. building with CMake
#   4. uploading the cache directory (if writing)
#
# Note: this file must be *sourced* not executed.

set -euo pipefail

# Configuration environment variables.
# TODO(#18238): Drop IREE_[READ,WRITE]_REMOTE_GCP_CCACHE after migration
IREE_READ_REMOTE_GCP_CCACHE="${IREE_READ_REMOTE_GCP_CCACHE:-1}"
IREE_WRITE_REMOTE_GCP_CCACHE="${IREE_WRITE_REMOTE_GCP_CCACHE:-0}"
IREE_READ_REMOTE_AZURE_CCACHE="${IREE_READ_REMOTE_AZURE_CCACHE:-0}"
IREE_WRITE_REMOTE_AZURE_CCACHE="${IREE_WRITE_REMOTE_AZURE_CCACHE:-0}"
IREE_READ_LOCAL_CCACHE="${IREE_READ_LOCAL_CCACHE:-0}"
IREE_WRITE_LOCAL_CCACHE="${IREE_WRITE_LOCAL_CCACHE:-0}"

if (( ${IREE_WRITE_REMOTE_GCP_CCACHE} == 1 && ${IREE_READ_REMOTE_GCP_CCACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_GCP_CCACHE' (${IREE_WRITE_REMOTE_GCP_CCACHE})" \
       " set without 'IREE_READ_REMOTE_GCP_CCACHE' (${IREE_READ_REMOTE_GCP_CCACHE})"
fi
if (( ${IREE_WRITE_REMOTE_AZURE_CCACHE} == 1 && ${IREE_READ_REMOTE_AZURE_CCACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_AZURE_CCACHE' (${IREE_WRITE_REMOTE_AZURE_CCACHE})" \
       " set without 'IREE_READ_REMOTE_AZURE_CCACHE' (${IREE_READ_REMOTE_AZURE_CCACHE})"
fi
if (( ${IREE_READ_REMOTE_GCP_CCACHE} == 1 && ${IREE_READ_REMOTE_AZURE_CCACHE} == 1 )); then
  echo "Can't have 'IREE_READ_REMOTE_GCP_CCACHE' (${IREE_READ_REMOTE_GCP_CCACHE})" \
       " set together with 'IREE_READ_REMOTE_AZURE_CCACHE' (${IREE_READ_REMOTE_AZURE_CCACHE})"
fi
if (( ${IREE_READ_REMOTE_AZURE_CCACHE} == 1 && [[ "${OSTYPE}" =~ ^msys ]] )); then
  # TODO(scotttodd): Windows support somehow?
  echo "Can't set 'IREE_READ_REMOTE_AZURE_CCACHE' (${IREE_READ_REMOTE_AZURE_CCACHE}) on Windows"
fi
if (( ${IREE_WRITE_LOCAL_CCACHE} == 1 && ${IREE_READ_LOCAL_CCACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_LOCAL_CCACHE' (${IREE_WRITE_LOCAL_CCACHE})" \
       " set without 'IREE_READ_LOCAL_CCACHE' (${IREE_READ_LOCAL_CCACHE})"
fi

if (( IREE_READ_REMOTE_GCP_CCACHE == 1 || IREE_READ_REMOTE_AZURE_CCACHE == 1 || IREE_READ_LOCAL_CCACHE == 1 )); then
  export IREE_USE_CCACHE=1
  export CMAKE_C_COMPILER_LAUNCHER="$(which ccache)"
  export CMAKE_CXX_COMPILER_LAUNCHER="$(which ccache)"
  ccache --zero-stats
  ccache --show-stats
else
  export IREE_USE_CCACHE=0
fi

if (( IREE_READ_LOCAL_CCACHE == 1 && IREE_WRITE_LOCAL_CCACHE == 0 )); then
  export CCACHE_READONLY=1
fi

if (( IREE_READ_REMOTE_GCP_CCACHE == 1 && IREE_READ_LOCAL_CCACHE == 0 )); then
  export CCACHE_REMOTE_ONLY=1
fi

if (( IREE_READ_REMOTE_AZURE_CCACHE == 1 && IREE_WRITE_REMOTE_AZURE_CCACHE == 0 )); then
  export CCACHE_READONLY=1
fi

if (( IREE_READ_REMOTE_GCP_CCACHE == 1 )); then
  export CCACHE_REMOTE_STORAGE="http://storage.googleapis.com/iree-sccache/ccache"
  if (( IREE_WRITE_REMOTE_GCP_CCACHE == 1 )); then
    set +x # Don't leak the token (even though it's short-lived)
    export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|bearer-token=${IREE_CCACHE_GCP_TOKEN}"
    set -x
  else
    export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|read-only"
  fi
fi

if (( IREE_READ_REMOTE_AZURE_CCACHE == 1 )); then
  # mkdir -p /mnt/azureblob
  # set +x
  # python3 ./build_tools/ccache/edit_fuse_connection.py ${{ secrets.AZURE_CCACHE_CONTAINER_KEY }}
  # set -x
  # blobfuse2 mount --allow-other --config-file=./build_tools/ccache/fuse_connection2.yaml /mnt/azureblob/
  # export CCACHE_DIR=/mnt/azureblob/ccache-container
fi
