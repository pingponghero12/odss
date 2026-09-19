# Benchmarks

Run the representative common-epoch SGP4 benchmark from an optimized installation:

```bash
python benchmarks/benchmark_sgp4.py --objects 10000
```

The benchmark reports wall-clock duration and propagated objects per second for one batch of OMM
records synchronized to a common epoch. It uses repeated deterministic elements so that it measures
propagation throughput without network or catalog-parsing variability.
