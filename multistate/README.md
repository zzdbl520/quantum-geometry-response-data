# Multistate quantum geometry and molecular response

Data, analysis scripts, and Supplementary Material for **Multistate Quantum
Geometry Distinguishes Physical Response from Electronic-State Switching**,
by Hui Li and Dongbo Zhao. Revision: **22 October 2026**.

This directory is the multistate study of C2, B2, and the H + O2 / OH + O
reaction geometries. The repository's root `data/` and `response_data.zip`
belong to the separate single-state response study.

## Start here

- [Supplementary Material (PDF)](supporting_information/JCP_Supporting_Information.pdf)
- [Supplementary Material (LaTeX)](supporting_information/JCP_Supporting_Information.tex)
- [Claim-to-file index](docs/CLAIM_EVIDENCE.md)
- [Data definitions and scientific scope](docs/DATA_DICTIONARY.md)
- [Reproduction and source-file provenance](docs/REPRODUCIBILITY.md)

| System | Current result | Numerical entry point |
|---|---|---|
| C2 | Two internal crossings, a separate target/complement crossing, and componentwise orbital-continuity checks | [Geometry points](evidence/C2/C2_SA6_global_transport_results_v1/analysis/transport_points.csv), [field limits](evidence/C2/C2_SA6_global_field_work_v2/analysis/zero_field_limits.csv) |
| B2 | Symmetry-allowed cusp and near-crossing response in an explicitly hybrid three-state model | [13-geometry response](evidence/B2/B2_stage2_combined_runs/stage2c_runs/B2_final_near_crossing_response.csv), [original versus hybrid pair flags](evidence/B2/B2_stage2_combined_runs/stage2c_runs/hybrid_pair_audit.csv) |
| Reaction | Two locally resolved scalar response maxima within the tested four-field window | [56 field pairs](evidence/HO2/manifold_pair_data.csv), [14-geometry summary](evidence/HO2/manifold_summary.csv), [retained failures](evidence/HO2/manifold_response_gate.json) |

## Reproduce the reported analyses

From this directory, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python scripts/verify_manifest.py
python scripts/reproduce.py
```

The scripts check finite-dimensional identities, recompute key molecular
quantities from the supplied tables, and regenerate `figures/C2.pdf`,
`figures/B2.pdf`, `figures/HO2.pdf`, and `figures/HO2_controls.pdf`.
Theoretical and molecular checks are written to `verification/`.
The exact library versions used for the released run are recorded in
[`verification/environment.json`](verification/environment.json).

The manifest check should be run before regenerating files: PDF metadata and
floating-point formatting can differ across environments even when the
underlying numerical results agree.

## What is established

- C2: the retained rank-three space contains both internal crossings. Its
  projected first-order dipole vanishes by parity. The external Delta/Pi
  crossing and the later optimized-orbital continuity failure are separate
  issues. Fixed-orbital controls are not inserted into relaxed response curves.
- B2: the model combines state-specific energies, direct common-orbital
  transition moments, and relaxed SA3 backgrounds. The direct/derivative
  difference remains in the records. The model cusp coefficient is not
  presented as a separately determined fully relaxed molecular value.
- Reaction: the lower envelopes at indices 5 and 9 exceed neighboring upper
  envelopes by 4.679532 and 0.211654 a.u. Full-matrix extrapolation still fails
  at 12 of 14 geometries, and the global scalar range test fails at index 18.

The release supports reanalysis of the distributed tables and regeneration
of the figures. It does not include every wavefunction checkpoint or a
complete clean-environment rerun of the original electronic-structure jobs.
Historical files under `evidence/` are preserved as supplied; their older
labels and absolute server paths are not the current reproduction protocol.
