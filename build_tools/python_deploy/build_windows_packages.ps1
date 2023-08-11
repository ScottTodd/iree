# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# One stop build of IREE Python packages for Windows. This presumes that
# dependencies are installed from install_windows_deps.ps1.

# Configure settings with script parameters.
param(
    [array]$python_versions=@('3.11'),
    [array]$packages=@('iree-runtime', 'iree-runtime-instrumented', 'iree-compiler'),
    [System.String]$output_dir
)

# Also allow setting parameters via environment variables.
if ($env:override_python_versions) { $python_versions = $env:override_python_versions };
if ($env:packages) { $packages = $env:packages };
if ($env:output_dir) { $output_dir = $env:output_dir };
# Default output directory requires evaluating an expression.
if (!$output_dir) { $output_dir = "${PSScriptRoot}\wheelhouse" };

$repo_root = resolve-path "${PSScriptRoot}\..\.."

# Canonicalize paths.
md -Force ${output_dir} | Out-Null
$output_dir = resolve-path "${output_dir}"

function run() {
  Write-Host "Using Python versions: ${python_versions}"

  $installed_versions_output = py --list | Out-String

  # Build phase.
  for($i=0 ; $i -lt $packages.Length; $i++) {
    $package = $packages[$i]

    echo "******************** BUILDING PACKAGE ${package} ********************"
    for($j=0 ; $j -lt $python_versions.Length; $j++) {
      $python_version = $python_versions[$j]

      if (!("${installed_versions_output}" -like "*${python_version}*")) {
        Write-Host "ERROR: Could not find python version: ${python_version}"
        continue
      }

      Write-Host ":::: Version: $(py -${python_version} --version)"
      switch ($package) {
          "iree-runtime" {
            clean_wheels iree_runtime $python_version
            build_iree_runtime $python_version
          }
          "iree-runtime-instrumented" {
            clean_wheels iree_runtime_instrumented $python_version
            build_iree_runtime_instrumented $python_version
          }
          "iree-compiler" {
            clean_wheels iree_compiler $python_version
            build_iree_compiler $python_version
          }
          Default {
            Write-Host "Unrecognized package '$package'"
            exit 1
          }
      }
    }
  }
}

function build_iree_runtime() {
  param($python_version)

  # TODO(scotttodd): remove debug logging
  Write-Host "build_iree_runtime; $python_version, $output_dir, $repo_root"
  $env:IREE_HAL_DRIVER_VULKAN = "ON"
  & py -${python_version} -m pip wheel -v -w $output_dir $repo_root/runtime/
}

function build_iree_runtime_instrumented() {
  param($python_version)
  Write-Host "build_iree_runtime_instrumented; $python_version"

  # TODO(scotttodd): Build
  # IREE_HAL_DRIVER_VULKAN=ON IREE_ENABLE_RUNTIME_TRACING=ON \
  # IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX="-instrumented" \
  # py -${python_version} -m pip wheel -v -w $output_dir $repo_root/runtime/
}

function build_iree_compiler() {
  param($python_version)
  Write-Host "build_iree_compiler; $python_version"

  # TODO(scotttodd): Build
  # local python_version="$1"
  # py -${python_version} -m pip wheel -v -w $output_dir $repo_root/compiler/
}

function clean_wheels() {
  param($wheel_basename, $python_version)
  Write-Host "clean_wheels; $wheel_basename :: $python_version"
  $cpython_version_string = "cp${python_version%.*}${python_version#*.}"

  # TODO(scotttodd): clean files
  # rm -f -v ${output_dir}/${wheel_basename}-*-${cpython_version_string}-*.whl
}

run
