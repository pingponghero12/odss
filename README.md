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

## Common-epoch SGP4 synchronization

Catalog elements retain their individual UTC epochs. Synchronize them to one absolute epoch before
using their Cartesian states together:

```python
simulation_epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "UTC")
synchronized = odss.synchronize_catalog_sgp4(catalog, simulation_epoch)
```

Each result preserves its original `OmmRecord`, records the signed propagation offset in SI seconds,
and provides a WGS-72 SGP4 state in TEME with position in metres and velocity in metres per second.
The scalar `propagate_omm_sgp4()` function provides the reference path; catalog synchronization uses
the accelerated batch interface. SGP4 error conditions are reported rather than silently returning
invalid states.

## Numerical propagation

Cascade is an optional numerical backend. Its conda-forge package is the preferred installation
because it includes the compatible C++ dependency stack:

```bash
conda install -c conda-forge cascade
```

Particle propagation uses SI values and requires explicit GCRS initial states:

```python
final_population = odss.propagate(
    initial_population,
    odss.CascadePropagationSpec(
        duration_s=5400.0,
        collisional_timestep_s=60.0,
    ),
)
```

The current numerical adapter uses point-mass Earth gravity only. It establishes the backend
boundary and must not be interpreted as the selected production SSO force model. Catalog targets
are first synchronized with SGP4 and transformed from TEME into the shared inertial simulation frame;
the architecture decision is recorded in `docs/adr/0001-mixed-propagation.md`.

## NASA breakup model

Fragmentation uses the external `nasa-sbm-py` package without copying its model into ODSS. The
current backend release supports Python 3.10 and should be installed separately from a sibling
checkout so its visualization-only dependencies do not replace ODSS's Astropy version:

```bash
python -m pip install "xarray<2026" "matplotlib<3.11"
python -m pip install --no-deps -e ../nasa-sbm-py
```

Generate the smallest characteristic-length cutoff needed by the study once, then derive larger
cutoffs without rerunning the stochastic model:

```python
key = odss.named_random_key(
    master_seed=2026,
    scenario_id=0,
    run_id=0,
    object_id=25544,
    stream_name="nasa_sbm",
)
fragments = odss.generate_explosion_fragments(
    parent_state,
    parent_properties,
    satellite_type="spacecraft",
    minimum_characteristic_length_m=0.01,
    random_key=key,
)
trackable_fragments = odss.select_fragments_by_size(fragments, 0.10)
```

Explosion and collision results contain an immutable `ParticlePopulation` plus characteristic
length, area-to-mass ratio, and delta-velocity components in SI units. The backend seed is derived
deterministically from the supplied ODSS random key and recorded with the result.
