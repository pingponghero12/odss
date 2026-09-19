# SSO force-model selection

ODSS delegates SSO force construction and integration to Cascade/heyoka. The available fidelity
terms are point-mass Earth gravity, J2/J3 zonal terms, C22/S22 tesseral terms, fitted atmospheric
drag, Sun and Moon gravity, and optional solar-radiation pressure.

Cascade's `simple_earth` model uses SI units in EME2000, with simulation time measured in seconds
from J2000 TT. ODSS rejects other frame labels and sets the backend clock from each population's
explicit epoch. Drag and radiation-pressure parameters are derived independently for every particle
from its cross-sectional area and mass.

The production model is not selected by enabling every term. `evaluate_force_model_sensitivity()`
records runtime and final-state differences against an explicit reference. Downstream studies should
also compare the scientific observables. The selected model should be the least expensive candidate
whose state and observable differences satisfy declared tolerances, keeping the complete initial
study within two days on 32 CPU cores.

Cascade 0.1.10's optional J4 expression returned a non-finite-state outcome in the validation orbit,
so it is deliberately not exposed in the initial fidelity ladder. This is preferable to presenting
an unvalidated higher-order term as additional accuracy.
