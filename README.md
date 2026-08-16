# odss

`odss` is a high-performance, reproducible framework for orbital-debris studies. Python provides
the study-definition and orchestration interface, while expensive processing belongs in the C++20
compiled core and future specialized scientific backends.

## Install

Python 3.10 or newer and a C++20 compiler are required.

```bash
python -m pip install .
```

For editable development installs, use `python -m pip install -e .`.

A local virtual environment can be created without changing the host installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest ruff pytest-cov
```

## Test

Run Python tests against the installed package:

```bash
python -m pytest

# For coverage
pytest --cov=odss --cov-report=term-missing
```

Build and run the C++ tests with a configured preset:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Direct CMake builds require development packages for pybind11 and GoogleTest. The `release`,
`asan`, and `ubsan` presets are available in the same way.
