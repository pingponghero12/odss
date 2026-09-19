# Monte Carlo execution and uncertainty

The statistical unit is one complete, independent breakup realization. A `MonteCarloRun` carries
the master seed, scenario ID, run ID, variant name, and backend thread allocation. Random keys use
the seed/scenario/run/object/stream address but deliberately exclude the variant name, so nominal
and sensitivity variants receive identical stochastic inputs for paired comparison.

Execution is serial by default. Setting `max_workers` uses independent worker processes, preserving
input order in the returned outcomes. The executor rejects allocations where
`max_workers * backend_threads_per_worker` exceeds the declared CPU budget and limits common
numerical-library thread pools inside worker processes. Production callbacks should also honor the
`backend_threads` value when configuring specialized backends.

Run-level Bernoulli probabilities use Wilson score intervals. Zero observed events use the exact
one-sided binomial upper bound, which approaches the 95% rule-of-three value `3 / N`. Means and
medians use a fixed-seed percentile bootstrap driven by ODSS's counter-based RNG. Pilot sample-size
helpers provide normal-approximation estimates; production run counts should be chosen separately
for each important observable and rounded conservatively.
