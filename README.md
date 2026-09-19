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

## Deterministic random values

Random values are addressed explicitly and can be requested in any order:

```python
key = odss.named_random_key(
    master_seed=2026,
    scenario_id=0,
    run_id=4,
    object_id=12,
    stream_name="fragment_mass",
)
sample = odss.uniform_01(key, draw_index=7)
```

The same address always produces the same value, independent of thread scheduling. A
`RunManifest` records the master seed, scenario ID, run ID, and RNG algorithm needed to identify
the run's stochastic inputs.

## Time and coordinate frames

Time conversion supports explicit UTC, TAI, and TT epochs, including UTC leap seconds. Cartesian
states can be explicitly transformed among TEME, GCRS, and ITRS using Astropy/ERFA:

```python
earth_orientation = odss.bundled_iers_a()
state_gcrs = odss.transform_state(
    state_teme,
    odss.ReferenceFrame("GCRS"),
    earth_orientation,
)
```

Frame conversion never downloads Earth-orientation data. The selected IERS Bulletin A or B file is
explicit and its `earth_orientation.metadata` can be included in a `RunManifest` input-assets list.

## OMM catalogs

CelesTrak-compatible OMM/JSON is parsed from exact bytes so its content hash can be retained with
explicit source and acquisition metadata:

```python
from pathlib import Path

import odss

catalog = odss.parse_omm_json(
    Path("catalog.json").read_bytes(),
    logical_name="catalog.json",
    source_uri="https://celestrak.org/...",
    acquired_at=acquisition_epoch,
)
```

Catalog unions select the latest epoch and then the greatest element-set number for each catalog
ID. `filter_sso()` requires an explicit precession-rate tolerance and compares the WGS-72 J2
secular node rate with the Sun's mean rate; it does not use an inclination/altitude box.
