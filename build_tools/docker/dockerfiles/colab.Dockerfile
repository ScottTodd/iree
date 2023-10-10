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

# Purge all Colab local runtime code we can find.
# We want Colab's dependencies but none of it's local environment configuration.
# RUN python -m pip uninstall -y google-colab
# RUN python -m pip uninstall -y jupyter_core jupyter-server nbconvert ipykernel
# RUN rm -rf /root/.jupyter/

# Install clean versions of Jupyter / nbconvert
# RUN python -m pip install jupyter_core nbconvert ipykernel

# python3 -m pip install --ignore-installed --quiet \
#   jupyter_core nbconvert ipykernel

# Purge all Colab local runtime code we can find.
# We want Colab's dependencies but none of it's local environment configuration.
# RUN jupyter server extension disable google.colab._serverextension
# RUN python -m pip uninstall -y google-colab
# RUN python -m pip uninstall -y jupyter-client jupyter-console jupyter_core jupyter-server jupyterlab-pygments jupyterlab-widgets
# RUN python -m pip uninstall -y nbconvert ipykernel
# RUN rm -rf /usr/local/etc/jupyter/

# Purge all paths that `jupyter --paths` returns
# -- config
# RUN rm -rf /.jupyter/
# RUN rm -rf /.local/etc/jupyter/
# RUN rm -rf /usr/etc/jupyter/
# RUN rm -rf /usr/local/etc/jupyter/
# RUN rm -rf /etc/jupyter/
# # -- data
# RUN rm -rf /.local/share/jupyter/
# RUN rm -rf /usr/local/share/jupyter/
# RUN rm -rf /usr/share/jupyter/
# # -- runtime
# RUN rm -rf /.local/share/jupyter/runtime/

# Install clean versions of Jupyter / nbconvert
# RUN python -m pip install jupyter_core nbconvert ipykernel

# TODO(scotttodd):
# replace 'colab_kernel_launcher' with 'ipykernel_launcher' in
# the files backing `jupyter kernelspec list --json` somehow?
