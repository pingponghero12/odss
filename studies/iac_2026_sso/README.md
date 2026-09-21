# SSO fragmentation study

This directory freezes the six principal scenarios: explosions and collisions at 500, 700, and
800 km. Every breakup is generated once down to 1 cm; 1, 5, and 10 cm populations are analyzed as
nested subsets. The preliminary campaign covers 7 days for runtime calibration. Production uses
the measured preliminary cost to select at most 64 realizations per scenario/model family and
propagates them for 365.25 days.

The nominal force model is J2 plus drag. Paired enhanced-force and drag-coefficient sensitivities
reuse the same run IDs at 700 km. The study reports proximity events, number/mass/kinetic-energy
flux, flux-derived model impact probabilities, decay/removal, escape, and a geometric maneuver-
demand proxy. It does not claim covariance-based operational collision probability.
Run files keep binary conjunction/maneuver indicators separately from the explicitly named Poisson
proxy so later Monte Carlo confidence intervals use the realization, rather than the fragment, as
their statistical unit. With no detected event, the stored minimum distance is censored at the
screening radius.

Cascade uses 120 s collisional steps batched 120 at a time. Each realization is single-threaded;
the campaign parallelizes independent realizations across at most 32 worker processes without
nested backend threading.

## Frozen inputs

The first real run downloads the official CelesTrak `weather`, `resource`, and `sar` groups in
OMM/JSON format. It archives the three untouched responses, acquisition UTC, URLs, byte counts,
SHA-256 hashes, and a deterministic merged catalog under `results/iac_2026_sso/input/`. Later runs
verify and reuse those exact bytes; they never silently refresh the population. The common
simulation epoch defaults to the recorded acquisition epoch, and the source commit is read from a
clean Git checkout. There are no executable `XX` placeholders.

OMM/JSON is deliberate: current catalog numbers can exceed the five-digit TLE limit. The merged
catalog is deduplicated and then filtered using the 5% nodal-precession SSO criterion.

Catalog OMM does not provide physical mass and area. Numerical target propagation therefore uses
an explicit 1000 kg / 10 m2 proxy. Impact probabilities are separately reported for 1, 5, 10, and
20 m2 reference areas. These assumptions must be stated in the paper and varied later if better
target metadata becomes available.

## Installation

Install ODSS and the NASA breakup wrapper in the normal virtual environment. Install Cascade from
conda-forge in the environment used for the campaign:

```bash
python -m pip install -e .
python -m pip install --no-deps -e ../nasa-sbm-py
conda install -c conda-forge cascade
```

Run the complete optional-backend validation before scientific execution:

```bash
odss-validate --include-optional-backends --output validation.json
```

To verify the entire real pipeline first, run one 10-minute explosion realization. This command
downloads and freezes the catalog on first use, runs NASA SBM and Cascade, and writes a NetCDF
result without prompts:

```bash
python studies/iac_2026_sso/smoke.py
```

## Preliminary campaign

First inspect the 24 planned files without requiring external inputs:

```bash
python studies/iac_2026_sso/preliminary.py --dry-run
```

Then run the approximately one-hour, 7-day calibration campaign. This is non-interactive; on its
first invocation it acquires and freezes the catalog automatically:

```bash
python studies/iac_2026_sso/preliminary.py --workers 32
```

## Production campaign

Production reads preliminary timing, scales it from 7 to 365.25 days, reserves a 25% runtime
margin, and chooses a run count within the 48-hour / 32-core budget:

```bash
python studies/iac_2026_sso/production.py \
  --pilot-summary results/iac_2026_sso/preliminary/campaign_summary.json \
  --workers 32
```

Use `--wall-hours` if the available allocation changes, or `--runs-per-family` to freeze a reviewed
count explicitly. Production stops if pilot timing says even one run per family exceeds the budget.
`--resume` skips existing immutable run files after an interrupted campaign.
The budget sizes the submitted workload; it is not a hard process kill, because terminating a
NetCDF write or Cascade integration would risk partial scientific output.

Each realization is one compressed NetCDF file. No trajectories are stored. The output directory
also contains the validation report, resolved campaign manifest, and timing summary needed to
justify the production sample count. This event-oriented layout is intended to remain compact
enough for a Zenodo data deposit. Publish the complete `results/iac_2026_sso/` directory so the
input snapshot, preliminary calibration, production results, and their manifests remain together.
The archived raw catalog responses and their hash manifest are part of that directory, so the
deposit contains the external scientific inputs needed to reproduce the published runs.
