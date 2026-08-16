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

Study-specific code belongs under `studies/`; the core library must not depend on the current IAC
study. Future backends may include Cascade, heyoka, SGP4, JAX, and CUDA, but this foundation neither
implements nor designs APIs for them.
