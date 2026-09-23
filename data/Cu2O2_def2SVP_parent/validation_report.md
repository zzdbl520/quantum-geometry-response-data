# Cu2O2 parent-CASCI(28e,16o) validation

## Audited outcome

The uploaded archive independently reproduces the strongest protocol decision:

`PARENT_CASCI2816_LOCAL_MINIMUM_VALIDATED`

- Input archive SHA256: `209a1dc8acb697e87f25e9c26eb163cff8c358d2a568a3dc06950af3ce298443`
- Protocol hash: `9c65abd54fc16cc5084d212083cdc6e9f102c461603cdc8d2e76adb9ccd54ce6`
- Production completeness: 15/15 states
- Preflight: PASS, with no errors or warnings
- Hard gates: all passed
- Preferred gates: both passed
- Independent quadratic reconstruction: PASS

The source CASSCF(8e,6o) minimum is at 1.756998 Å and the strictly nested
parent CASCI(28e,16o) minimum is at 1.754195 Å. The absolute displacement is
0.002803 Å, only 5.61% of the 0.05 Å geometry spacing.

## Response data

| R(O-O) / Å | Source alpha_xx / a.u. | Parent alpha_xx / a.u. | Parent - source / a.u. | Relative correction | h vs 2h discrepancy | Energy vs dipole discrepancy |
|---:|---:|---:|---:|---:|---:|---:|
| 1.70 | 108.623153 | 108.728344 | 0.105190 | 0.0968% | 0.0040% | 0.0192% |
| 1.75 | 108.375300 | 108.503181 | 0.127880 | 0.1180% | 0.0041% | 0.0259% |
| 1.80 | 108.514748 | 108.663637 | 0.148890 | 0.1372% | 0.0094% | 0.0276% |

The parent-space correction is small and smooth. The fitted curvature changes
from 154.9202 to 154.2479 a.u., a relative change of only -0.434%. Thus the
larger CI space produces primarily a gentle upward displacement rather than a
qualitative reshaping of the response curve.

## Numerical and orbital diagnostics

| Diagnostic | Observed | Acceptance level |
|---|---:|---:|
| Maximum parent h/2h relative discrepancy | 9.400e-5 | <= 3.0e-2 |
| Maximum parent energy/dipole relative discrepancy | 2.760e-4 | <= 3.0e-2 |
| Maximum abs(S^2) | 1.728e-13 | <= 1.0e-6 |
| Maximum active-electron error | 5.826e-13 | <= 1.0e-8 |
| Minimum target-parent singular value | 0.95120694 | preferred >= 0.95 |
| Minimum adjacent-geometry parent singular value | 0.98118935 | preferred >= 0.95 |
| Minimum same-geometry field-continuity singular value | 0.99999768 | >= 0.990 at 2h |
| Source-active nesting singular value | approximately 1.00000000 | >= 0.999999 |

The target-representation value is closest to its preferred threshold at point
009 under the +2h field. However, its paired -2h value differs by less than
1e-9, and the values change smoothly with geometry. Together with essentially
unit field continuity, this behavior supports a stable parent-space
construction rather than a field-induced orbital selection change.

## Recommended scientific claim

The result supports the following claim:

> The local Cu-Cu-axis polarizability minimum near R(O-O) = 1.75 Å is
> preserved when the wave function is diagonalized in a strictly nested
> CAS(28e,16o) parent CI space constructed from independently field-relaxed
> CASSCF(8e,6o) orbitals.

It does not establish an orbital-optimized CASSCF(28e,16o), a basis-set limit,
dynamic-correlation convergence, or a double-shell limit.

## Recommended main-text insertion

> To assess active-space sensitivity without allowing a redefinition of the
> active manifold, we additionally diagonalized a strictly nested
> CASCI(28e,16o) parent space on the independently field-relaxed
> CASSCF(8e,6o) orbitals. The fitted Cu2O2 alpha_xx minimum is retained and
> moves by only 0.0028 Å, from R(O-O) = 1.7570 to 1.7542 Å. The parent-space
> correction is smooth and amounts to only 0.097-0.137% of alpha_xx, while the
> fitted curvature changes by 0.43%, showing that the minimum is not removed
> by this substantial enlargement of the CI space.

For a shorter main-text sentence:

> A strictly nested parent CASCI(28e,16o)//CASSCF(8e,6o) validation preserves
> the Cu2O2 alpha_xx minimum, shifting its fitted position by only 0.0028 Å
> (1.7570 to 1.7542 Å).

The existing abstract need not be lengthened; this validation is strongest as
a Results/SI robustness statement rather than a new headline claim.

## Recommended SI methods paragraph

> Active-space sensitivity of the Cu2O2 response minimum was examined at
> R(O-O) = 1.70, 1.75, and 1.80 Å using fields F_x = 0, +/-h, and +/-2h, with
> h = 2.5e-4 a.u. For each geometry and field, the complete six-dimensional
> active span of the independently converged CASSCF(8e,6o)/def2-SVP source
> state was retained exactly. Ten orthonormal combinations were promoted only
> from the source-core space by maximizing their representation of the Cu 3d
> and O 2p IAO target after projection of the source-active component. The
> resulting strictly nested parent space contains 28 electrons in 16 orbitals
> and has an M_S = 0 determinant dimension of 14,400. Fixed-orbital singlet
> CASCI calculations were then performed without orbital optimization or
> canonicalization. Polarizabilities were obtained from the five-point energy
> stencil. All 15 states converged, and the field-step, energy-dipole,
> spin-purity, electron-count, orbital-nesting, and orbital-continuity tests
> satisfied the prespecified acceptance criteria.

## Figure captions

### Compact main-text/inset figure

> Active-space validation of the Cu2O2 polarizability minimum. Symbols are
> five-point finite-field alpha_xx values at R(O-O) = 1.70, 1.75, and 1.80 Å;
> curves are quadratic interpolants. Open gray symbols denote
> CASSCF(8e,6o)/def2-SVP and filled red symbols denote the strictly nested
> fixed-orbital parent CASCI(28e,16o) calculation. Triangles and vertical
> dotted lines mark the fitted minima. Enlargement of the CI space shifts the
> minimum by only 0.0028 Å.

### Supporting-information figure

> Strictly nested parent-CI validation of the Cu2O2 response minimum. (a)
> Five-point finite-field alpha_xx values and quadratic interpolants for the
> source CASSCF(8e,6o) and fixed-orbital parent CASCI(28e,16o) calculations.
> The fitted minima occur at R(O-O) = 1.7570 and 1.7542 Å, respectively. (b)
> Parent-space correction to alpha_xx; percentage labels give the correction
> relative to the source value. The small, smooth correction and 0.0028 Å
> vertex displacement demonstrate that the minimum is preserved upon strict
> enlargement of the CI space.

## Placement recommendation

- Use the two-panel figure in the Supporting Information with the full methods
  paragraph and numerical table.
- Use the compact single-panel version only as an inset or if the main text has
  room for a dedicated active-space robustness check.
- In all manuscript locations, retain the phrase "fixed-orbital parent CASCI"
  or an equivalent qualifier; do not call this a converged CASSCF(28e,16o)
  calculation.
