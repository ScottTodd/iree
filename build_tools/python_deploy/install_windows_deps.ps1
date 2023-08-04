# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs dependencies on Windows necessary to build IREE Python wheels.

$PYTHON_VERSIONS = @(
  "311",
  # "310",
  # "39"
)

$PYTHON_INSTALLER_URLS = @(
  "https://www.python.org/ftp/python/3.11.2/python-3.11.2-amd64.exe",
  # "https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe",
  # "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
)

for($i=0 ; $i -lt $PYTHON_VERSIONS.Length; $i++) {
  $PYTHON_VERSION = $PYTHON_VERSIONS[$i]
  $PYTHON_INSTALLER_URL = $PYTHON_INSTALLER_URLS[$i]
  Write-Host "-- Installing Python ${PYTHON_VERSION} from ${PYTHON_INSTALLER_URL}"

  # Note: Multiple paths are valid. We just check one format - this is brittle.
  #   C:\Python39\python.exe
  #   C:\Program Files\Python39\python.exe
  #   C:\Users\[NAME]\AppData\Local\Programs\Python\Python39\python.exe
  $PYTHON_PATH = "C:\\Program Files\\Python${PYTHON_VERSION}"
  $PYTHON_EXE = "${PYTHON_PATH}\\python.exe"

  if (!(Test-Path -Path ${PYTHON_EXE} -PathType Leaf)) {
    $DOWNLOAD_ROOT = "$env:TEMP/iree_python_install"
    $DOWNLOAD_FILENAME = $PYTHON_INSTALLER_URL.Substring($PYTHON_INSTALLER_URL.LastIndexOf("/") + 1)
    $DOWNLOAD_PATH = "${DOWNLOAD_ROOT}/$DOWNLOAD_FILENAME"

    # Create download folder as needed.
    md -Force ${DOWNLOAD_ROOT} | Out-Null

    Write-Host "  Downloading $PYTHON_INSTALLER_URL -> $DOWNLOAD_PATH"
    curl $PYTHON_INSTALLER_URL -o $DOWNLOAD_PATH

    Write-Host "  Running installer: $DOWNLOAD_PATH"
    # https://docs.python.org/3/using/windows.html#installing-without-ui
    & $DOWNLOAD_PATH /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
  } else {
    Write-Host "  Python version already installed. Not reinstalling."
  }

  Write-Host "  Python version $PYTHON_VERSION installed:"
  & $PYTHON_EXE --version
  & $PYTHON_EXE -m pip --version

  Write-Host "  Installing system pip packages"
  & $PYTHON_EXE -m pip install --upgrade pip
}

Write-Host "*** All done ***"
