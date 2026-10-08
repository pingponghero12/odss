# SSO fragmentation study

This directory freezes the six principal scenarios: explosions and collisions at 500, 700, and
800 km. Every breakup is generated once down to 5 cm; 5 and 10 cm populations are analyzed as
nested subsets. The preliminary campaign uses ten one-hour realizations per family to estimate
runtime and run-level uncertainty. Production uses the measured preliminary cost to select at most
20 realizations per scenario family over 12 simulated hours.

The force model is J2 plus drag. The study reports proximity events, number/mass/kinetic-energy
flux, flux-derived model impact probabilities, final orbital classification, escape, and a
geometric maneuver-demand proxy. It does not claim covariance-based operational collision probability.
Run files keep binary conjunction/maneuver indicators separately from the explicitly named Poisson
proxy so later Monte Carlo confidence intervals use the realization, rather than the fragment, as
their statistical unit. With no detected event, the stored minimum distance is censored at the
screening radius.

Cascade uses 120 s collisional steps processed one at a time. Each realization is single-threaded;
the campaign parallelizes independent realizations across worker processes without nested backend
threading. Workers are restarted between balanced waves so native backend memory is released.

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

First inspect the 60 planned files without requiring external inputs:

```bash
python studies/iac_2026_sso/preliminary.py --dry-run
```

Then run the one-hour, ten-realization-per-family preliminary campaign. This is non-interactive; on its
first invocation it acquires and freezes the catalog automatically:

```bash
python studies/iac_2026_sso/preliminary.py --workers 11
```

## Production campaign

Production reads preliminary timing, reserves a 25% runtime margin, and chooses a run count within
the 48-hour compute allocation:

```bash
python studies/iac_2026_sso/production.py \
  --pilot-summary results/iac_2026_sso/preliminary/campaign_summary.json \
  --runs-per-family 20 --workers 11 --wall-hours 48
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

## Manuscript figures and tables

The completed manuscript campaign has 120 files: 20 runs per scenario over 12 h.
The one-hour pilot is separate and is not pooled into these statistics. Regenerate
the manuscript assets from a complete campaign with:

```bash
python -m pip install 'matplotlib>=3.7,<3.11'
python -m studies.iac_2026_sso.prepare_paper results/iac_2026_sso/production
```

The generator rejects incomplete or mixed ensembles and records the run-file
hashes. To use a larger study later, preserve the old outputs, complete a new
campaign with its own manifest, and pass its directory instead. Review manuscript
conclusions after changing the data. See `docs/paper_sso/README.md` for paper-build
instructions, orbital visualization, and local data-preservation notes.

The archive's `removal_fraction` and `reentry_fraction` are disjoint final
osculating-perigee classes, not physical removal or observed re-entry counts.
Their sum is the final bound-fragment fraction at or below 200 km. Breakup kicks
can already create low perigees, and particles are not removed at this boundary.
The paper therefore labels this quantity **final low-perigee fraction** and does
not attribute it solely to drag or infer a retention time history.
