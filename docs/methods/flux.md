# Target-centered flux estimator

For each target, ODSS counts unique fragment encounters with closest-approach distance no greater
than a configurable sampling radius `R`. Over a time bin of duration `dt`, the sampling area is
`A = pi R^2` and the reported fluxes are:

```text
number flux         = sum(1)                  / (A dt)  [m^-2 s^-1]
mass flux           = sum(fragment mass)      / (A dt)  [kg m^-2 s^-1]
kinetic-energy flux = sum(0.5 mass speed^2)   / (A dt)  [W m^-2]
```

The speed is the debris-target relative speed at TCA. Results are emitted for every target, time
bin, and characteristic-size bin, including zero-count bins. The result contains no particle-backend
state, so a future phase-space-density estimator can produce the same `FluxResult` structure.

The sampling radius is an estimator parameter, not a physical collision radius. Studies should
evaluate multiple radii and look for a stable flux range. A larger radius increases the number of
samples but becomes less local; a smaller radius is more local but may have excessive counting
noise.
