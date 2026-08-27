# References

Citations for the algorithms, thresholds and baselines this implementation
relies on. Fuller discussion is on the project wiki; this file is the version
that travels with the code.

## Data

- **Landsea, C. W., and J. L. Franklin, 2013:** Atlantic hurricane database
  uncertainty and presentation of a new database format. *Mon. Wea. Rev.*,
  **141**, 3576–3592. [doi:10.1175/MWR-D-12-00254.1](https://doi.org/10.1175/MWR-D-12-00254.1)

  HURDAT2 format, and the source for the quantisation `emulate_working_fix()`
  applies: best tracks are recorded to 5 kt, 1 mb and 0.1° lat/lon. Also
  establishes that a best track is a *smoothed* post-storm analysis — the
  property the working/final split exists to keep out of model inputs.

- **Torn, R. D., and C. Snyder, 2012:** Uncertainty of tropical cyclone
  best-track information. *Wea. Forecasting*, **27**, 715–729.
  [doi:10.1175/WAF-D-11-00085.1](https://doi.org/10.1175/WAF-D-11-00085.1)

  The calibration target for `WorkingTrackNoise`. Satellite-only intensity
  uncertainty near 10–12 kt and pressure 7–12 mb — roughly double the scope
  defaults. Also intensity-dependent, which the scalar-RMS emulator cannot
  express. See `WorkingTrackNoise.from_literature()`.

- **Sampson, C. R., and A. J. Schrader, 2000:** The Automated Tropical Cyclone
  Forecasting System (version 3.2). *Bull. Amer. Meteor. Soc.*, **81**,
  1231–1240. The ATCF a-deck/b-deck format.

- **Hersbach, H., and Coauthors, 2020:** The ERA5 global reanalysis.
  *Q. J. R. Meteorol. Soc.*, **146**, 1999–2049.
  [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

## Physics and thresholds

- **Emanuel, K. A., 1995:** *J. Atmos. Sci.*, **52**, 3969–3976.
  [doi:10.1175/1520-0469(1995)052<3969:SOTCTS>2.0.CO;2](https://doi.org/10.1175/1520-0469\(1995\)052%3C3969:SOTCTS%3E2.0.CO;2)
- **Bister, M., and K. A. Emanuel, 1998:** Dissipative heating and hurricane
  intensity. *Meteor. Atmos. Phys.*, **65**, 233–240.
  [doi:10.1007/BF01030791](https://doi.org/10.1007/BF01030791)

  Together these are the potential-intensity algorithm `features.potential_intensity()`
  stands in for. Citing 1995 alone understates it — the operational `pcmin` code
  is the open-cycle formulation with dissipative heating.

- **Kaplan, J., and M. DeMaria, 2003:** *Wea. Forecasting*, **18**, 1093–1108.
  [doi:10.1175/1520-0434(2003)018<1093:LCORIT>2.0.CO;2](https://doi.org/10.1175/1520-0434\(2003\)018%3C1093:LCORIT%3E2.0.CO;2)

  `RI_THRESHOLD_KT = 30.0`. The 95th percentile of Atlantic 24 h intensity
  changes, not a physical discontinuity.

- **NHC track forecast cone:** <https://www.nhc.noaa.gov/aboutcone.shtml> —
  `CLIMATOLOGICAL_CONE_NM` provenance, and the 2026 elliptical-cone trial.

## Predictability

- **Emanuel, K., and F. Zhang, 2016:** On the predictability and error sources of
  tropical cyclone intensity forecasts. *J. Atmos. Sci.*, **73**, 3739–3747.
  [doi:10.1175/JAS-D-16-0100.1](https://doi.org/10.1175/JAS-D-16-0100.1)

  Intensity error growth over the first few days is dominated by **initial
  intensity error** — what the noise emulator perturbs, at the leads the
  promotion gates score. This is why recalibration is a prerequisite.

- **Emanuel, K., and F. Zhang, 2017:** *J. Atmos. Sci.*, **74**, 2315–2324.
  [doi:10.1175/JAS-D-17-0008.1](https://doi.org/10.1175/JAS-D-17-0008.1)
  Inner-core moisture matters as much as the wind field. Not represented in
  `FEATURE_NAMES`.

- **DeMaria, M., C. R. Sampson, J. A. Knaff, and K. D. Musgrave, 2014:**
  *Bull. Amer. Meteor. Soc.*, **95**, 387–398.
  [doi:10.1175/BAMS-D-12-00240.1](https://doi.org/10.1175/BAMS-D-12-00240.1)
- **Gall, R., and Coauthors, 2013:** The Hurricane Forecast Improvement Project.
  *Bull. Amer. Meteor. Soc.*, **94**, 329–343.
  [doi:10.1175/BAMS-D-12-00071.1](https://doi.org/10.1175/BAMS-D-12-00071.1)

  Track and intensity are not the same problem; `DEFAULT_THRESHOLDS` should not
  treat them symmetrically.

## Verification

- **Diebold, F. X., and R. S. Mariano, 1995:** *J. Bus. Econ. Statist.*, **13**,
  253–263. [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- **Coroneo, L., and F. Iacone, 2024:** Testing for equal predictive accuracy
  with strong dependence. [arXiv:2409.12662](https://arxiv.org/abs/2409.12662)

  Why `aggregate_by_storm()` exists: with dependent losses the DM test loses
  power entirely and can spuriously reject a correct null.

- **Gneiting, T., and A. E. Raftery, 2007:** *J. Amer. Statist. Assoc.*, **102**,
  359–378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437) —
  the CRPS kernel form used in `crps_ensemble`.
- **Hamill, T. M., 2001:** Interpretation of rank histograms for verifying
  ensemble forecasts. *Mon. Wea. Rev.*, **129**, 550–560.
  [doi:10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2](https://doi.org/10.1175/1520-0493\(2001\)129%3C0550:IORHFV%3E2.0.CO;2)

  A U-shaped histogram is not uniquely underdispersion.

- **NHC Forecast Verification:** <https://www.nhc.noaa.gov/verification/> —
  official 48 h Atlantic track error was 45.4 n mi (2024) and 53.4 n mi (2025).

## Prior art

- **Lam, R., and Coauthors, 2023:** *Science*, **382**, 1416–1421.
  [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) (GraphCast)
- **Bi, K., and Coauthors, 2023:** *Nature*, **619**, 533–538.
  [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3) (Pangu-Weather)
- **Kurth, T., and Coauthors, 2023:** FourCastNet: accelerating global
  high-resolution weather forecasting. *PASC '23*.
  [doi:10.1145/3592979.3593412](https://doi.org/10.1145/3592979.3593412)
- **Price, I., and Coauthors, 2025:** Probabilistic weather forecasting with
  machine learning. *Nature*, **637**, 84–90.
  [doi:10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9) (GenCast)

  GenCast is independent confirmation of the §4.6 policy: trained on ERA5, then
  fine-tuned on operational HRES-fc0 analyses, with both weight sets released.

- **Song, Y., P. Dhariwal, M. Chen, and I. Sutskever, 2023:** Consistency models.
  *ICML*, PMLR **202**, 32211–32252.
  [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)

  The alternative to load shedding if the MERIDIAN step budget binds.

- **Raissi, M., P. Perdikaris, and G. E. Karniadakis, 2019:** *J. Comput. Phys.*,
  **378**, 686–707. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

  Governing equations enter through the loss, not the forward pass — the §3.6
  correction `models/pinn.py` implements.
