# Changelog

## 1.2.5

- Moved the panel (b) decomposition legend above the plotting frame and set it
  on one horizontal line so that no data curve is obscured.

## 1.2.4

- Reduced the interpanel spacing in the two-panel main-text figure.
- Replaced the ambiguous `M1`/`M2` annotations by the explicit labels
  `Peak I` and `Peak II`; these denote the first and second validated
  polarizability maxima along the IRC, not mechanism names.

## 1.2.3

- Compressed the main publication figure to a focused two-panel 1-by-2 layout:
  validated response topology in panel (a) and its external/internal mechanism
  decomposition in panel (b).
- Moved the late-path SA2 solution-change diagnostic to the standalone
  `HO2_SI_late_path_control.pdf/png` control figure; it is no longer presented
  as a third main-text mechanism panel.

## 1.2.2

- Rearranged the publication three-panel figure into an inverted-triangle
  layout: total response and mechanism decomposition on top, with the SA2
  solution-change control centered below.
- Replaced the signed, source-13-centered Cartesian arc coordinate by one
  common dimensionless IRC progress coordinate, `source_index/(N_IRC-1)`,
  matching the 0--1 convention used for the source reaction profile.  The
  late-path panel is now explicitly identified as a local IRC window.

## 1.2.1

- Added local-extremum and reaction-topology gates that remain meaningful when
  an unrelated endpoint control narrowly misses the global plateau tolerance.
- Added `ROBUST_NOT_MAXIMUM` and adjacent nonoverlapping-envelope orderings;
  these establish, without fitting, that the late-path response rises through
  sources 16--18 rather than forming the apparent SA2 third maximum.
- Expanded the default target set to sources 5, 8--10, and 16--17 so both
  validated maxima and the rejected alternatives receive explicit statuses.
- Added the publication figure `HO2_response_mechanisms.pdf/png`, separating
  the rank-2 response, its external/internal contributions, and the SA2
  solution-change control.

## 1.2.0

- Split validation into a strict full-operator extrapolation gate and an
  invariant observable-mechanism gate; a noisy off-diagonal matrix element no
  longer vetoes a separately stable physical ground-state extremum.
- Added a fixed common-field plateau audit over
  `2.5e-4,5.0e-4,7.5e-4,1.0e-3` a.u. for `A_00`, the retained-manifold term,
  reconstructed ground response, trace response, and trace metric.
- Added median, lower/upper envelope, relative range, and per-source plateau
  failure columns.
- Added conservative extrema: a robust maximum requires its lower raw-field
  envelope to exceed both neighboring upper envelopes.
- Added `internal-manifold-generated` classification when the external
  response is not a maximum but the reconstructed response is.
- Added target status reporting for sources 5, 10, and 16.
- Restored the validated intermediate field window as the production default;
  the smaller v1.1.1 fields remain useful as a subtractive-cancellation audit.

## 1.1.1

- Added the physical nondegenerate lowest-state reconstruction
  `alpha_0=A_00+2 sum_j |M_0j|^2/(E_j-E_0)` and reports its external and
  internal-manifold components separately.
- Added reconstructed-response extrema and their internal-response fraction.
- Added explicit per-source numerical failure reasons to the JSON gate.
- Added fit-stability and dipole-fit checks to the numerical gate.
- Replaced the exploratory manifold field window by the smaller perturbative
  window `6.25e-5,1.25e-4,1.875e-4,2.5e-4` a.u.; existing largest points are
  reused automatically.

## 1.1.0

- Added signed-field SA4 calculations for a locally homed rank-2 response
  manifold selected from four candidate roots.
- Added exact many-electron subspace homing and polar transport into the
  zero-field gauge.
- Added covariant dipole and polarizability operators, the non-Abelian
  projector metric, and an effective response-energy operator by congruence.
- Added stable-rank diagnostics so replacement of an unused fourth SA4 root
  does not invalidate a continuous rank-2 response manifold.
- Added numerical gates, CSV/JSON summaries, a four-panel diagnostic figure,
  and regression tests for the manifold linear algebra.

## 1.0.1

- Fixed `ss_qgeom_summary.csv` generation when the first selected geometry is
  unbracketed but later geometries contain local-extremum diagnostics.
- Added a regression test for heterogeneous bracketed/unbracketed summary rows.

## 1.0.0

- Added an isolated protocol-validation work tree; original SA2 checkpoints
  and field files are never overwritten.
- Added directly transported state-specific ground-state CASSCF(13,9).
- Added equal-weight SA4-CASSCF(13,9) zero-field transport and four-state
  subspace continuity audits.
- Added an 80-task SS finite-field pilot over source windows 4--6, 9--11,
  and 15--18 with the validated large-field set.
- Replaced the naive three-point maximum gate by a conservative uncertainty
  envelope, logarithmic-slope sign change, and 12-scheme topology audit.
- Added an idempotent source-21--30 extension route using geometries already
  present in the H2COMBUSTION reaction-1 dataset.
