# Scientific validation reports

Run the deterministic core report with:

```bash
odss-validate --output validation.json
```

When Cascade, heyoka, or nasa-sbm-py are installed, include their integration cases with
`--include-optional-backends`. Unavailable optional backends are reported as skipped; they are not
silently omitted. Every numerical check records its measured value, reference, units, absolute
error, and tolerance so a failure can be investigated directly.

The automated cases cover a known SGP4 state, explicit frame/time handling, common-epoch agreement,
continuous TCA, analytical flux and radius convergence, analytical impact risk, deterministic
end-to-end processing, NASA SBM repeatability, propagation tolerance convergence, and Cascade versus
brute-force screening.

The SGP4 and documented frame cases provide practical external implementation references. MASTER
and ORDEM products are not bundled because they are external environment-model outputs rather than
small redistributable fixtures. A literature cloud-density comparison also requires a fully
specified, redistributable reference case before it can become an automated numerical acceptance
test; no values are invented in its absence.
