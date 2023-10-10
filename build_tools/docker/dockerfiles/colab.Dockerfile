# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running IREE's Colab notebooks.

# https://research.google.com/colaboratory/local-runtimes.html
FROM us-docker.pkg.dev/colab-images/public/runtime

ARG PYTHON_VERSION=3.10
RUN apt-get install -y "python${PYTHON_VERSION}-venv"
