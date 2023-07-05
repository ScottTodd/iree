#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR dialect markdown document postprocessor for website generation.

The markdown files generated by `iree-tblgen` and `mlir-tblgen` are generic,
but we want to process them through the same static site generator (mkdocs)
as the rest of our website. This script allows us to make certain style and
structure changes to each file in a directory.

Usage (typically invoked by generate_extra_files.sh, edits files in-place):
  postprocess_dialect_docs.py ${WEBSITE_DOCS_BUILD_DIRECTORY}
"""

import argparse
import fileinput
import pathlib
import re


def main(args):
    directory = args.directory
    files = list(pathlib.Path(directory).iterdir())

    # Set certain headings to depth 5, lining up with the table-of-contents
    # (toc) max depth setting of 4 (`toc_depth: 4`).
    #
    # Skipping heading levels, e.g.
    #     # Heading 1
    #     <-- no Heading 2 here! -->
    #     ### Heading 3
    # is usually discouraged, but we aren't getting much value from treating
    # these as subsections and we don't want them showing up in the rendered
    # table of contents.
    #
    # (We could also use another form of emphasis like bolt/italics instead)
    #
    # Example:
    #
    #   # 'foo' Dialect
    #   ## Operation definition
    #   ### 'group bar' Ops
    #   #### `foo.bar.baz` (foo::BarBazOp)
    #   ##### Attributes:                       <--- No change needed
    #   ##### Operands:                         <--- No change needed
    #   ##### Results:                          <--- No change needed
    #   ### `foo.ungrouped` (foo::UngroupedOp)
    #   #### Attributes:                        <--- Change this heading level
    #   #### Operands:                          <--- Change this heading level
    #   #### Results:                           <--- Change this heading level
    #   ## Attribute definition
    #   ### FooAttr
    #   #### Parameters:                        <--- Change this heading level
    with fileinput.input(files=files, inplace=True) as f:
        for line in f:
            line = re.sub(r"^#### Attributes", "##### Attributes", line)
            line = re.sub(r"^#### Parameters", "##### Parameters", line)
            line = re.sub(r"^#### Operands", "##### Operands", line)
            line = re.sub(r"^#### Results", "##### Results", line)

            print(line, end="")

    # Add frontmatter to the start of each file.
    frontmatter = """
---
hide:
  - tags
tags:
  - MLIR
---

"""
    for filename in files:
        with open(filename, "r+") as f:
            original_content = f.read()
            f.seek(0, 0)
            f.write(frontmatter + original_content)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialect doc postprocessor.")

    parser.add_argument(
        "directory",
        help="Dialect docs directory to edit in-place.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
