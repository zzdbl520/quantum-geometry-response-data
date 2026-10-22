# Data definitions and interpretation

Atomic units are used for fields, dipoles, response, and metric values unless
a column explicitly specifies another unit. Energies labelled `Eh` are in
hartree; distances labelled `A` or `Angstrom` are in angstrom.

## Reaction pair table

File: `evidence/HO2/manifold_pair_data.csv`.

| Column | Meaning in the current manuscript |
|---|---|
| `source_index` | Index in the supplied reaction geometry set; plotting coordinate xi = index / 57. |
| `s_cart_A` | Cartesian path coordinate with zero at source index 13; not the normalized plotting coordinate. |
| `field_au` | Positive magnitude defining an opposite-field pair. |
| `minus_selected_indices`, `plus_selected_indices` | Selected two-state subsets of the four candidate states at the negative and positive fields. |
| `minus_anchor_sv_min`, `plus_anchor_sv_min` | Smallest singular value of the overlap with the local zero-field rank-two anchor. |
| `minus_plus_sv_min` | Smallest singular value between the selected opposite-field spaces. |
| `dipole_00`, `dipole_01`, `dipole_11` | Entries of D(F) = [H_tilde(+F) - H_tilde(-F)] / (2F). Historical names retained. For H = H0 - F mu at fixed orbitals, D = -mu. |
| `alpha_ij` | Entries of A(F) = -[H_tilde(+F) + H_tilde(-F) - 2H0] / F^2. |
| `external_ground_alpha_au` | A00, the aligned background. Includes orbital response; not a purely external-state sum in optimized CASSCF. |
| `internal_manifold_ground_alpha_au` | I = 2 abs(D01)^2 / [E1(0)-E0(0)]. The gap is `internal_gap_zero_Eh` in the geometry summary. |
| `reconstructed_ground_alpha_au` | A00 + I, formed at each field before taking a median. |
| `metric_ij` | Overlap-based finite-field projector-metric matrix entries. Their zero-field extrapolation is subject to separate convergence criteria. |
| `trace_Delta_geom_Eh` and principal operator diagnostics | Historical exploratory values. The current reaction conclusions do not claim a converged response-weighted energy spectrum. |

`plateau_*` in `manifold_summary.csv` describes medians/ranges over four
specified field magnitudes. The historical prefix does not by itself mean
the zero-field limit has converged. The total median need not equal the sum
of component medians. A positive neighboring-envelope margin establishes a
sampled local ordering within the tested window, not a statistical confidence
interval or a continuous-coordinate peak location.

## B2 hybrid construction

File: `evidence/B2/B2_stage2_combined_runs/stage2c_runs/B2_final_near_crossing_response.csv`.

| Column | Meaning |
|---|---|
| `stage1b_X_minus_A_gap_Eh` | Signed gap between the state-specific X and A curves used in the model. |
| `mu_direct_perpendicular_au` | Direct transition moment from a common-orbital SA3 calculation. |
| `mu_relaxed_transport_perpendicular_au` | Magnitude of the aligned field-optimized derivative. |
| `orbital_response_renormalization_ratio` | Ratio of the preceding derivative magnitude to the direct moment. This label does not establish a unique orbital-only decomposition. |
| `alpha_*_final_au`, `metric_*_final_au` | Hybrid model quantities, combining direct-moment inverse-gap terms with SA3 finite-field backgrounds. |
| `cusp_slope_direct_perpendicular_au` | Cusp coefficient of the model constructed with the direct coupling at its state-specific crossing. |
| `status` | Hybrid-protocol status; not passage of the earlier direct/derivative comparison. |

The pair audit preserves `original_status`, `original_reasons`, and
`diagnostic_flags`. Different quantities may have different normalization
(trace versus mean per retained state). Use the explicit column names; do
not substitute a trace for a scalar branch response.

## C2 target versus geometry frame

SA6 defines the geometry continuation frame. The response projector spans
X plus both Delta components and has rank three. The external Delta/Pi
crossing closes a gap for this target even when the six-state frame remains
continuous. A rank-six aggregate flag and a rank-three component flag can
therefore differ. Only the current componentwise plot selection defines
the accepted response curves.

The common-energy moment operator, actual projector metric of a split group,
and overlap-derived finite-field metric are distinct objects with different
conditions. Historical exploratory arithmetic-denominator matrices must not
be read as exact off-diagonal projector metrics; SI Section S3 explains why.
