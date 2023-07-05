# API bindings

API bindings allow for programmatic use of IREE's compiler and runtime
components. The core IREE project is written in C[^1], allowing for API bindings
to be written in a variety of other languages.

Each page in this section covers one of the official or unofficial language
bindings:

Language | Compiler API? | Runtime API? | Published packages?
-------- | ------------ | ----------- | ------------------
[C/C++](./c-api.md) | :white_check_mark: Supported | :white_check_mark: Supported | :x: Unsupported
[Python](./python.md) | :white_check_mark: Supported | :white_check_mark: Supported | :white_check_mark: Supported
Java | :x: Unsupported | :grey_question: Experimental | :x: Unsupported
JavaScript | :grey_question: Experimental | :grey_question: Experimental | :x: Unsupported
Julia | :grey_question: Experimental | :grey_question: Experimental | :x: Unsupported
Rust | :x: Unsupported | :grey_question: Experimental | :grey_question: Experimental

!!! question - "Something missing?"

    Want to use another language? Looking for something specific out of one of
    those already listed?

    We welcome discussions on our
    [communication channels](../../index.md#communication-channels) and
    contributions on [our GitHub page](https://github.com/openxla/iree)!

[^1]: with some C++ tools and utilities
