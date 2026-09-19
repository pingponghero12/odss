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

Run the intentionally small brute-force conjunction correctness baseline with:

```bash
python benchmarks/benchmark_reference_screening.py --debris 200 --targets 200
```

This computes exact continuous interior closest-approach events for every debris-target pair under
the local constant-velocity model. It is a golden reference measurement, not a production
throughput target.

With Cascade installed, run the production local screening benchmark with:

```bash
python benchmarks/benchmark_cascade_screening.py --debris 10000 --targets 2000
```

The input is a deterministic LEO-like local population. For profiling before optimization, run the
same command through `python -m cProfile -s cumulative`.

Compare the SSO force-model fidelity ladder with Cascade installed:

```bash
python benchmarks/benchmark_force_models.py --objects 1000 --duration-s 86400
```

The benchmark reports runtime and final-state differences from the explicitly configured reference
model. Production selection must also check convergence of the downstream observables.

Measure process-level Monte Carlo scaling while keeping each run single-threaded:

```bash
python benchmarks/benchmark_monte_carlo.py --runs 32 --workers 1 2 4 8
```

The benchmark reports throughput and speedup for deterministic independent runs. Real production
measurements should use the same worker/thread allocation with the representative scientific run.
