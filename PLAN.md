# Implementation plan

How this codebase maps onto Scope v2.1, what is real, and what has to be built
next. Read `README.md` first for orientation.

---

## 1. Design stance

The scope describes an ML system, but the parts most likely to produce a wrong
forecast are not the models. They are the boundaries: which data may be read
when, which artifact is allowed to be promoted, what happens when a feed is
late. v2 had defects in exactly those boundaries — same-cycle GFS that does not
exist yet, final best-track used as an input, promotion on the test set, a
single-model retrain silently invalidating the ensemble generator.

So this implementation puts the effort there. Every boundary rule from v2.1 is a
function that raises, called at the point where violating it would be a one-line
change. The models are honest implementations with correct shapes and latent
contracts, but they are the replaceable part.

The practical consequence: `uv sync` without the torch extra gives you a
package where all the policy logic is importable and testable. Model code is
opt-in.

---

## 2. Build order

Each layer was built and tested before the next, since each depends on the
guarantees below it.

**Layer 1 — time and geometry.** `time_utils`, `geo`. Synoptic arithmetic and
the t-6 selection rule. Everything downstream assumes cycle t never reads the
cycle named t, so that is enforced here rather than remembered everywhere else.

**Layer 2 — data contracts.** `data/sources` (role registry), `data/besttrack`
(working vs final, noise emulator), `data/availability` (what has published),
`data/features` (one code path per flavor), `data/splits` (storm-wise +
chronological). This layer is where the train/serve policy is mechanised.

**Layer 3 — training policy.** `training/curriculum` (Stage A/B),
`training/promotion` (validation-only gates, test-set budget),
`training/orchestrator` (wave scheduling), `training/triggers` (cascade).

**Layer 4 — tracking.** `tracking/tags` (validated tag set),
`tracking/registry` (MLflow-optional, model-set pinning).

**Layer 5 — inference.** `inference/scheduler` (timeline, gating, load
shedding), `inference/cycle` (execution, degraded modes),
`inference/postprocess` (products).

**Layer 6 — monitoring and metrics.** `monitoring/skew`, `monitoring/drift`,
`metrics/track`, `metrics/probabilistic`.

**Layer 7 — models.** Seven PyTorch builders, all returning `(module, spec)`.

---

## 3. Design decisions worth stating

**Availability is a queryable model, not an assumption.** `AvailabilityOracle`
is a protocol; `LatencyOracle` implements it from the registry's latency table,
with per-source overrides and outage injection. Every timing test is written
against a configured oracle rather than a mocked clock, so an outage scenario is
three lines and a replay harness for a real season is a second implementation of
the same protocol.

**Degradation is always flagged, never silent.** Every fallback path appends to
`CycleOutput.flags` and surfaces in the dissemination payload: `vitals=estimated`,
`nwp_stale=12h`, `meridian_fallback:RuntimeError`. A forecaster who cannot tell
a full-ensemble cycle from a climatological-fallback cycle will eventually treat
both the same way, and the second is much weaker.

**Opportunistic feeds are distinguished from optional ones.** Aircraft recon and
microwave overpasses are absent from most cycles by nature. Flagging their
absence would mark every cycle degraded and destroy the flag's meaning, so
`InputStatus.opportunistic` separates "normally missing" from "unexpectedly
missing".

**The fusion weight floor.** Both the learned fusion layer and the fallback
`fusion_weights` clamp weights above a floor. Mid-season there may be only a
handful of verifying storms; a model zeroed out on three cases is unrecoverable
for the rest of the season.

**The test set is metered.** `TestSetBudget` rate-limits evaluations and demands
a written justification. §4.4 reserves 2023-2025 for an unbiased comparison
against NHC consensus, and that reservation is only worth anything if looks are
counted.

**Latent coupling is a signature.** `latent_signature()` hashes the Group 1
version set; derived models record the signature they trained against, and
`pin_set` refuses a set whose signatures disagree. This is what makes "retrain
the affected model only" impossible to do accidentally.

---

## 4. What is synthetic, and what replacing it involves

| Component | Status | To productionise |
|---|---|---|
| Storm archive | Synthetic, seeded | Parse HURDAT2 into `Track`; keep `TrackQuality.FINAL` |
| Working track | Emulated from final | Ingest real a-deck/b-deck/TC-Vitals; run `recalibrate_from_pairs` |
| Gridded fields | Synthetic, dual-flavor | GRIB2 readers for GDAS/GFS and ERA5 into `GriddedFields` |
| Satellite | Not implemented | GOES-18/19 storm-relative crops into the CNN channel stack |
| Potential intensity | SST/OHC/shear regression | Full Emanuel (1995) from thermodynamic soundings |
| Cone radii | Placeholder table | Current-season NHC error percentiles |
| Appendix B thresholds | Provisional | Re-baseline against the current NHC verification report |
| Models | Untrained | Stage A on ERA5, Stage B on GDAS, per `configs/curriculum.yaml` |

The synthetic generator's ERA5/GDAS offset (`synthetic.GDAS_BIAS`) is
deliberate and load-bearing for the tests. It is not a claim about the real
offset between those analysis systems.

---

## 5. Open items

Citations for everything below are in `docs/references.md`.


**Sample size.** Roughly 10-15k synoptic fixes exist across the full archive.
For a 6-layer transformer and a diffusion model that is thin. Stage A on ERA5
helps, but the honest mitigation is aggressive augmentation, storm-relative
coordinates, and being ready to conclude the transformer is not the right
capacity for this dataset. Worth measuring before committing GPU-months.

**Ensemble dispersion at recurvature.** MERIDIAN conditioned on AEOLUS latents
will tend to underdisperse precisely where the distribution is bimodal.
`build_diffusion(extra_conditioning_dim=...)` exists so raw fields or GEFS/EPS
perturbations can be added to the conditioning vector. The rank histogram and
spread-skill ratio in verification are how you find out whether that was enough,
and both should be tracked per-lead from the first backtest.

**Absolute skill targets.** The Appendix B numbers (<75 nm at 48h and similar)
may already trail current NHC performance. The beat-rate gates carry the real
claim because they measure against a live baseline; the absolute thresholds
should be re-baselined against the current verification report before anyone
treats them as a bar.

**Noise-emulator recalibration is a prerequisite, not a refinement.** Emanuel and
Zhang (2016) find intensity error growth over the first few days is dominated by
initial-intensity error — exactly what `emulate_working_fix()` perturbs, at
exactly the leads the promotion gates score. The scope's 5 kt / 3 mb defaults are
roughly half the published estimates (Torn and Snyder 2012), and the real error
is intensity-dependent, which the scalar-RMS emulator cannot express.
`WorkingTrackNoise.from_literature()` exists to measure the sensitivity before
real paired data arrives.

**Two structural gaps in the feature set, both intensity-side.** No inner-core
moisture (Emanuel and Zhang 2017 find it matters as much as the wind field;
`rh700_pct` is an environmental area mean). No ocean feedback — SST and OHC are
static daily values persisted from the previous day, so a storm's own cold wake,
a first-order limit on its own intensification, is nowhere in the system.

**Track and intensity thresholds should not share a table.** Intensity skill has
improved far more slowly than track skill, so a beat rate on intensity is a claim
about a near-static baseline close to an intrinsic predictability limit. The
absolute limits in `DEFAULT_THRESHOLDS` also need re-deriving outright: NHC's
official 48 h Atlantic track error was 45.4 n mi in 2024 and 53.4 n mi in 2025,
against a 90 nm production threshold.

**Consistency distillation beats load shedding if the diffusion budget binds.**
Song et al. (2023): one-step generation by design, multistep still available,
distillable from an already-trained diffusion model — a graded response to time
pressure rather than dropping members. Evaluate before relying on the shedding
path in production.

**Load shedding is planned but not implemented downstream.** The scheduler
selects the reduced profile and flags it, but `run_cycle` does not yet reduce
the requested member count. Wiring `CyclePlan.load_shed` through to the ensemble
generator is a small change and should happen before any live use.

---

## 6. Test suite map

| File | Covers |
|---|---|
| `test_time_utils.py` | Synoptic arithmetic; t-6 selection; t-12 fallback |
| `test_sources.py` | Role assignment; operational guard |
| `test_besttrack.py` | Working/final separation; emulator statistics; recalibration |
| `test_availability.py` | Publication timing; outages; opportunistic feeds |
| `test_scheduler.py` | Cycle timeline; vitals gating; load shedding; deadlines |
| `test_curriculum.py` | Stage A/B ordering; flavor rules; deployability |
| `test_promotion.py` | Validation-only gates; manual gate; test-set budget |
| `test_registry.py` | Versioning; model-set pinning; rollback; MLflow degradation |
| `test_triggers.py` | Trigger conditions; cascade; nightly-latent suppression |
| `test_orchestrator.py` | Wave schedules; dependency handling; failure isolation |
| `test_skew.py` | ERA5T audit; alert thresholds; windowing |
| `test_drift.py` | Feature drift; reference flavor; validation-loss trigger |
| `test_metrics.py` | Track/intensity errors; beat rate; DM test; CRPS; spread-skill |
| `test_features_splits.py` | Dual-flavor parity; flavor guards; split leakage |
| `test_cycle.py` | End-to-end cycle; every degraded mode; extrapolation |
| `test_postprocess.py` | Cone construction and fallback; PDF; landfall; RI |
| `test_tags.py` | Tag validation; v2.1 additions |
| `test_synthetic.py` | Generator reproducibility and statistics |
| `test_models.py` | Architecture shapes and latent contracts (needs torch) |
