# Decay, escape, and impact-risk observables

Decay diagnostics use osculating two-body elements calculated from explicit geocentric Cartesian
states. ODSS reports changes in semimajor axis, perigee altitude, and apogee altitude. Final states
are classified using configurable altitude boundaries as retained, removal, reentry, or escape.
These labels are simulation diagnostics; a perigee crossing a boundary does not model atmospheric
demise.

Escape means non-negative Earth-centered specific orbital energy:

```text
energy = speed^2 / 2 - mu / radius
```

Impact risk is derived from number flux rather than waiting for rare literal collisions. For target
reference area `A`, ODSS integrates `lambda = A * integral(flux dt)` and reports
`model_impact_probability = 1 - exp(-lambda)` under an explicitly stated rare independent-impact
approximation. This quantity is not an operational covariance-based collision probability.
