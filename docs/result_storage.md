# Event-oriented result storage

ODSS stores selected run outputs in a normalized SQLite database. Every row carries the study,
scenario, and run identity associated with its canonical run manifest. The stored scientific rows
cover scalar run summaries, conjunction events, target-wise flux bins, and maneuver-demand output.

```python
odss.write_run_result("results.sqlite", result)
restored = odss.read_run_result("results.sqlite", result.identity)
```

Writes are transactional and an existing run identity is never silently replaced. Run workers
should return immutable `RunResult` values to one coordinating writer instead of making many worker
processes contend for the same database.

Internal propagation states are intentionally absent. A study may separately retain selected debug
trajectories, but normal Monte Carlo execution does not persist every integration step.
