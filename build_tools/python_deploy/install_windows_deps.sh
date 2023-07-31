#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs dependencies on Windows necessary to build IREE.
#
# Usage:
#   bash install_windows_deps.sh

set -eox pipefail

# if [[ "$(whoami)" != "root" ]]; then
#   echo "ERROR: Must setup deps as root"
#   exit 1
# fi

PYTHON_SPECS=(
  # 311@https://www.python.org/ftp/python/3.11.2/python-3.11.2-amd64.exe
  310@https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe
  # 39@https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
)

for python_spec in "${PYTHON_SPECS[@]}"; do
  python_version="${python_spec%%@*}"
  url="${python_spec##*@}"
  echo "-- Installing Python $python_version from $url"
  # python_path="/Library/Frameworks/Python.framework/Versions/$python_version"
  # python_exe="$python_path/bin/python3"
  python_path="C:\\Python$python_version"
  python_exe="$python_path\\python.exe"
  #  C:\Python311\python.exe

  echo "  $python_path"
  echo "  $python_exe"

  # Install Python.
  if ! [ -x "$python_exe" ]; then
    echo "  $python_exe does not exist"

    echo "  TEMP: $TEMP"

    package_basename="$(basename $url)"
    download_path="$TEMP/iree_python_install/$package_basename"
    mkdir -p "$(dirname $download_path)"
    echo "Downloading $url -> $download_path"
    curl $url -o "$download_path"

    # https://docs.python.org/3/using/windows.html#installing-without-ui
    echo "Installing $download_path"
    $download_path /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    # cmd.exe /c start $download_path /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    # installer -pkg "$download_path" -target /
  else
    echo ":: Python version already installed. Not reinstalling."
  fi

  # echo ":: Python version $python_version installed:"
  # $python_exe --version
  # $python_exe -m pip --version

  # echo ":: Installing system pip packages"
  # $python_exe -m pip install --upgrade pip
done

echo "*** All done ***"
