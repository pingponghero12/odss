# Architecture

```text
Python study/orchestration layer
        |
        v
Python package API
        |
        v
compiled backend interfaces
        |
        v
C++20 implementation / external scientific backends
```

Python is intended for study configuration and orchestration. Expensive processing should occur in
compiled or vectorized backends. The scientific API should favor immutable specifications and
functional composition, and every implementation layer should remain independently unit-testable.

Core scientific values are implemented in C++20 and exposed as immutable Python types. Particle
populations store each numerical field in its own contiguous array so later compiled kernels can
process them as structure-of-arrays data without changing the public model.

Experiment identity is derived in the Python orchestration layer from schema-tagged canonical JSON.
Floating-point values use exact hexadecimal text, and SHA-256 provides experiment and study
identities. Provenance metadata is explicit and contains no implicit timestamps or process state.

Deterministic random values use a stateless Philox4x64-10 counter generator. The counter contains
the scenario, run, object, and draw-block IDs; the key contains the master seed and stream ID.
Python stream names are converted to stable 64-bit IDs with domain-separated SHA-256. This makes
random access independent of execution order and thread scheduling. Run manifests record the
run-level RNG identity but not individual object, stream, or draw coordinates.

Study-specific code belongs under `studies/`; the core library must not depend on the current IAC
study. Future backends may include Cascade, heyoka, SGP4, JAX, and CUDA, but this foundation neither
implements nor designs APIs for them.
