# Mixed catalog and fragment propagation

## Decision

Catalog objects remain OMM records until a simulation epoch is selected. ODSS evaluates them in a
batch with SGP4 at that epoch, transforms the resulting TEME states into the simulation's explicitly
selected inertial frame, and then numerically propagates both targets and breakup fragments with one
Cascade simulation and one shared force model.

The source OMM records, catalog identifiers, acquisitions, and synchronization epoch remain
provenance data alongside the propagated particle population. They must not be inferred back from
Cartesian states.

## Why

Cascade applies one symbolic dynamics system to every particle. Per-particle runtime parameters can
change coefficients, but Cascade cannot consume an external SGP4 ephemeris as the trajectory of only
some particles. Keeping targets on SGP4 while fragments use numerical dynamics would therefore need
a separate continuous cross-trajectory screening implementation. It would also make the two groups
use different dynamical assumptions after the common epoch.

Starting all objects from a common epoch and propagating them with the same model gives Cascade
continuous trajectories for conjunction detection. SGP4 remains the authoritative conversion from
catalog mean elements to the initial target states; it is not treated as a force model.

## Scientific constraints

- SGP4 output is TEME. It must be transformed explicitly before it is combined with fragment states.
- Cascade's point-mass adapter currently accepts GCRS states only and preserves SI units.
- Cross-sectional area is converted to Cascade's spherical particle radius as
  `sqrt(area_m2 / pi)`.
- The point-mass dynamics used by the adapter is an integration spike, not the selected production
  SSO force model.
- Catalog identity and physical properties are separate inputs. The OMM data does not provide mass
  or cross-sectional area.

## Rejected alternative

Segmenting time, refreshing target states with SGP4 at every segment boundary, and numerically
propagating only fragments would introduce target trajectory discontinuities and leave conjunctions
near boundaries dependent on the chosen segment size. It is not selected without an accuracy study
showing that this additional complexity is necessary.
