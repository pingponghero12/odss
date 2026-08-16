# AGENTS.md

## Project purpose

Project name is `odss` - orbital debris simulation system

This repository implements a high-performance, reproducible framework for
orbital-debris studies.

The public-facing workflow is Python-based and should allow studies to be
defined compositionally using a clean, mostly functional API.

Computationally expensive operations should use optimized compiled backends,
primarily modern C++20. Existing high-quality scientific libraries such as
Cascade, heyoka, SGP4 implementations, and nasa-sbm-py should be reused rather
than reimplemented without a concrete reason.

The initial scientific application is an SSO fragmentation study, but the core
library must remain generic and must not contain IAC-2026-specific assumptions.

`CONTEXT.md` contains background discussion and scientific rationale. It is
reference material, not the implementation specification. The current PR/task
requirements override `CONTEXT.md` if there is any conflict.


## Core priorities

Priorities, in order:

1. Scientific and numerical correctness.
2. Reproducibility.
3. Computational performance.
4. Clear and minimal architecture.
5. Developer usability.

Do not sacrifice correctness for speed, but performance is a first-class
requirement.

The eventual production study must be designed to complete in approximately
7 days or less on a 32-core CPU.

Avoid implementations that are obviously incompatible with this target.


## Scope discipline

Implement only the requested PR.

Do not proactively implement functionality planned for later PRs.

Do not introduce abstractions only because they might theoretically be useful
later.

Do not build custom implementations of functionality already provided by an
appropriate external scientific library unless the PR explicitly requires it.

If a future requirement affects an interface, leave a clean extension point
rather than implementing the future feature.

Keep changes focused and reviewable.


## Architecture

Keep the repository structure simple, shallow, and explicit. Add architectural
layers only when a current requirement needs them.

Use snake_case consistently for project-defined Python and C++ modules,
functions, variables, files, targets, and similar identifiers unless an
external API requires another convention.

Put C++ template declarations on their own line immediately above the
function, class, or variable they declare.

Do not mention PR or roadmap numbers in source comments or docstrings. Explain
the lasting scientific or architectural reason instead.

### Python

Python is the user-facing orchestration and study-definition layer.

Prefer:

- immutable specifications;
- pure functions;
- explicit inputs and outputs;
- composition;
- type hints;
- small modules.

Avoid:

- mutable global state;
- setter-heavy APIs;
- large stateful `Simulation` objects;
- hidden configuration;
- unnecessary inheritance hierarchies;
- generic `dict[str, Any]` scientific APIs.

Study configuration should be ordinary Python code, not YAML or JSON.

Configuration may be serialized internally for provenance and reproducibility.


### C++

Use modern C++20 for performance-critical functionality.

Prefer:

- RAII;
- value semantics;
- contiguous memory;
- simple ownership;
- explicit types;
- batch processing;
- data-oriented layouts where beneficial;
- compiler-vectorizable loops.

Avoid:

- unnecessary heap allocation in hot loops;
- per-element virtual dispatch;
- unnecessary copies;
- deeply nested abstraction in numerical kernels;
- premature template complexity.

## Performance

The complete production workload must eventually fit within approximately one
week on a 32-core CPU.

For computationally significant code:

1. Establish correctness first.
2. Create a representative benchmark.
3. Profile before performing substantial optimization.
4. Optimize measured bottlenecks.
5. Verify numerical results after optimization.

Do not optimize trivial setup or orchestration code.

Avoid unnecessary Python loops over large particle populations.

Prefer batched calls into compiled backends.

Avoid nested parallelism that oversubscribes hardware, for example 32 outer
workers each spawning 32 internal worker threads.

Release benchmarks must use optimized builds.

Do not enable unsafe floating-point optimizations such as `-ffast-math` by
default.


## Testing

Testing is mandatory at every implementation layer.

Every new scientific or numerical component must include unit tests.

Every bug fix should include a regression test when practical.

Tests must be deterministic.

Use small synthetic cases with known answers wherever possible.

The project will eventually contain:

- unit tests;
- integration tests;
- numerical reference tests;
- deterministic regression tests;
- end-to-end tests;
- performance benchmarks.

Do not replace correctness tests with benchmarks.

Do not make CI depend on large external datasets.


## Reproducibility

Scientific results must be reproducible.

Never introduce unseeded stochastic behavior.

Thread scheduling must not silently alter stochastic inputs.

Do not use hidden timestamps, random UUIDs, or process-global RNG state as part
of scientific calculations.

External scientific inputs must eventually be identifiable by hashes and
provenance metadata.


## Numerical code

Use SI units internally unless an interface explicitly states otherwise.

Names should communicate units where ambiguity is possible, for example:

- `position_m`
- `velocity_m_s`
- `mass_kg`
- `area_m2`
- `duration_s`

Do not silently mix coordinate frames, epochs, units, or reference systems.

Scientific assumptions must be explicit in public specifications rather than
hidden inside implementation code.


## Dependencies

Prefer mature, maintained scientific libraries instead of duplicating their
functionality.

Keep dependencies minimal.

Do not add a dependency merely to avoid writing a few lines of straightforward
code.

Large or specialized scientific backends should be optional where practical.

Do not copy code from external projects into this repository unless explicitly
required and license-compatible.


## Code quality

Before completing a task:

- build the affected targets;
- run relevant C++ tests;
- run relevant Python tests;
- run formatting/linting expected by the repository;
- inspect warnings;
- summarize what changed and what was tested.

Do not silence compiler warnings without understanding them.

Do not weaken existing tests merely to make a change pass.


## Documentation

Public interfaces require concise documentation.

Document scientific meaning and assumptions, not obvious syntax.

For architectural decisions with meaningful tradeoffs, prefer a short ADR or
design note rather than extensive comments scattered through implementation.

Comments should explain why, not restate what the code does.


## When uncertain

Do not invent scientific methodology.

Do not silently make major architecture decisions outside the PR specification.

For ambiguous low-level implementation details, choose the smallest solution
consistent with:

- the current PR;
- this file;
- existing repository conventions;
- future backend extensibility.

`CONTEXT.md` may be consulted for additional rationale when needed.
