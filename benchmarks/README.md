# Benchmarks

Run the representative common-epoch SGP4 benchmark from an optimized installation:

```bash
python benchmarks/benchmark_sgp4.py --objects 10000
```

The benchmark reports wall-clock duration and propagated objects per second for one batch of OMM
records synchronized to a common epoch. It uses repeated deterministic elements so that it measures
propagation throughput without network or catalog-parsing variability.

With Cascade installed, run the representative numerical SSO propagation benchmark from an
optimized environment:

```bash
python benchmarks/benchmark_propagation.py --objects 10000 --duration-s 5400
```

It reports wall time and simulated object-seconds per wall second for a deterministic circular SSO
population under the adapter's point-mass spike dynamics.
