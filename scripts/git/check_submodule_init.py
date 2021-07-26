# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys


def run():
  # TODO: no-op if not a git repository

  output = os.popen("git submodule status")
  submodules = output.readlines()

  for submodule in submodules:
    if (submodule.strip()[0] == "-"):
      print(
          "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
          % (submodule.split()[1]))
      sys.exit(1)


if __name__ == "__main__":
  run()
