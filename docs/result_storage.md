# Xarray and NetCDF result storage

Each Monte Carlo realization is represented as an `xarray.Dataset` and stored in one NetCDF file.
This follows the repository's scientific-file workflow: run files remain independently readable,
portable, and easy to aggregate with xarray.

```python
dataset = odss.run_result_dataset(result)
odss.write_run_result("run_000001.nc", result)
restored = odss.read_run_result("run_000001.nc")
```

The dataset contains scalar coordinates for the study, scenario, and run identity, plus the
canonical manifest JSON. Flux uses dense `target`, `time_bin`, and `size_bin` dimensions.
Conjunctions and maneuver-triggering encounters use separate event dimensions because their
counts vary between runs. Scientific arrays carry explicit unit attributes.

Parallel workers should write separate run files. Publication uses a temporary file followed by
an atomic filesystem operation, and an existing run file is never silently overwritten. Internal
trajectory time steps are intentionally excluded from this event-oriented result product.

The in-memory dataset can also be written with xarray's `to_zarr()` when the optional Zarr
dependencies are installed; NetCDF is the supported on-disk format provided by odss itself.
