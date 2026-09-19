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

Astronomical time and frame conversions are explicit Python boundary operations backed by
Astropy/ERFA. Supported time scales are UTC, TAI, and TT; supported geocentric frames are TEME,
GCRS, and ITRS. Frame transforms require an explicit IERS Bulletin A or B table and disable
automatic downloads. The table's content hash and size are exposed as provenance metadata. Frame
conversion remains separate from propagation and force models.

OMM/JSON ingestion produces immutable GP records with explicit UTC epochs and TEME/SGP4 metadata.
Catalog inputs retain exact content hashes, source URIs, and caller-supplied acquisition epochs.
Unions resolve duplicate catalog IDs by element epoch and element-set number, rejecting ambiguous
ties. Sun-synchronous filtering uses the WGS-72 J2 secular nodal-precession approximation and an
explicit caller tolerance rather than a geometric orbit box.

Catalog synchronization uses the maintained `sgp4` implementation with the WGS-72 gravity model.
The scalar path is the reference operation and the catalog path uses its accelerated satellite-array
API. Both produce SI-valued TEME states at one explicit absolute epoch while retaining each original
OMM record and its element epoch. The signed propagation offset is stored on every synchronized
object. Heyoka SGP4 is not used because its current lack of deep-space propagation would make this
general catalog operation invalid for GEO objects.

Study-specific code belongs under `studies/`; the core library must not depend on the current IAC
study. Other propagation backends remain outside the catalog synchronization interface.
