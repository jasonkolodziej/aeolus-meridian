# Train/serve consistency

Scope v2.1 §4.6. The single most consequential change from v2, and the reason
several modules exist in the shape they do.

## The problem

v2 trained on ERA5 reanalysis and final HURDAT2 best-track, then served on GFS
analyses and real-time ATCF fixes. Those are different distributions, and the
difference is not small:

- **ERA5 vs GDAS.** Different assimilation systems, different model cores,
  different bias characteristics. ERA5 also has ~5 days of latency, so it can
  never be an operational input — the fallback to GFS described in v2 was not a
  fallback, it was the permanent operating condition.
- **Final vs working best-track.** HURDAT2 is reanalysed after the season using
  aircraft, scatterometer and satellite data that did not exist in real time,
  then smoothed. A model trained on it learns to rely on precision that is
  unobtainable at inference.

A model fitted on the first distribution and served the second is being asked to
extrapolate at exactly the moment it matters.

## The policy

Three rules, each mechanised:

1. **ERA5 pretrains; GDAS serves.** Stage A learns representation from four
   decades of homogeneous reanalysis. Stage B re-seats the weights on GDAS
   analyses from 2015-present. Only Stage B output is registrable.
2. **Final best-track is a label, never an input.** Model inputs come from
   working-quality fixes. For historical training where no working track was
   archived, the emulator degrades the final track by measured error.
3. **One code path per derived feature.** Shear is computed by the same function
   in both flavors, so any measured gap is attributable to data rather than code.

## Where each rule lives

| Rule | Function | Raises |
|---|---|---|
| No reanalysis in the cycle | `sources.assert_not_operational` | `OperationalUseError` |
| No final track as input | `besttrack.assert_input_safe` | `ValueError` |
| No flavor mixing | `features.assert_flavor` | `FlavorMismatchError` |
| Curriculum ends operational | `curriculum.assert_deployable` | `CurriculumError` |
| No pretrain weights registered | `registry.register` | `RegistryError` |
| No pretrain metrics gate promotion | `promotion.evaluate_promotion` | `PromotionError` |

Six independent checks for one policy is deliberate. Each guards a different
entry point, and any single one being bypassed still leaves the others.

## The noise emulator

`emulate_working_fix` degrades a final fix by:

- an isotropic 2-D position displacement with the measured RMS (Rayleigh
  magnitude, so per-axis sigma is `rms/sqrt(2)`) — not independent lat/lon
  jitter, which would understate error near the poles and impose a preferred axis
- Gaussian intensity and pressure error
- quantisation to 0.1 degree and 5 kt, matching operational reporting

Starting constants are 15 nm, 5 kt and 3 mb. These are documented placeholders.
`recalibrate_from_pairs` measures the real values from archived
working/final pairs and should replace them before Stage B is taken seriously.

Note that quantisation means a small perturbation sometimes leaves a position
unchanged. That is faithful to the real product, not a bug —
`test_quantisation_can_leave_a_position_unchanged` pins the behaviour.

## The audit

A policy that cannot detect its own failure is an assertion. `monitoring/skew`
implements §4.6.3: about five days after each cycle, ERA5T becomes available;
the deterministic stack is re-run on it and compared to what the operational
GDAS-driven run produced.

A rolling 14-day mean 48h track delta above 15 nm, or intensity delta above
4 kt, raises an alert and triggers a Stage B re-fine-tune — Stage B only, since
the pretrained representation is still valid; what has moved is the operational
distribution.

The monitor returns `None` below 8 samples rather than a noisy estimate. With a
handful of storms a fortnight, a two-sample mean would fire the retrain trigger
on one unusual case.
