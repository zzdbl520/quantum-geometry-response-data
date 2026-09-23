# Data for *Quantum Geometry Governs Molecular Response Reversals*

This repository contains derived response data for H2, N2, Cr2, F2, OsO4,
and the ligand-supported dicopper–dioxygen complex. It accompanies the article
and Supporting Information (SI) of the same title.

## Contents

- [`data/`](data/): all 69 numerical data and audit files from the verified
  release, unpacked for browsing. CSV files contain response curves and fitted
  positions; JSON and text files document the field and branch checks.
- [`response_data.zip`](response_data.zip): the complete 109-file release,
  including those same data files, analysis and plotting scripts, source
  figures, a SHA-256 manifest, and a reproducibility guide. This is the
  self-contained version to download for the checks below.

| Main-text result | Data to start with | SI detail |
| --- | --- | --- |
| H2 maximum (Fig. 1) | `data/H2_qgeom_derived.csv`, `data/H2_qgeom_log_slopes.csv` | Fig. S1, Table S2 |
| N2 comparison | `data/N2_qgeom_derived.csv` | Fig. S2, Table S2 |
| Cr2 minimum and maximum (Fig. 2) | `data/Cr2_aug_global40_derived.csv`, `data/Cr2_aug_global40_extrema.csv`, field and geometry sensitivity CSVs | Fig. S3, Tables S3–S4 |
| OsO4 compression minimum (Fig. 3) | `data/OsO4_def2TZVP/OsO4_CAS24_def2TZVP_CI_recovery/OsO4_def2TZVP_qgeom_summary.csv` | Fig. S5, Table S5 |
| F2 primary maximum and transverse feature (Fig. 4) | `data/F2_stage2f_publication_data.csv`, `data/F2_stage2f_physical_audit.json` | Fig. S4 |
| Dicopper path and response (Figs. 5–6) | `data/Cu2O2_def2TZVP/DEF2TZVP_25POINT_RESPONSE_POINTS.csv`, `data/Cu2O2_def2TZVP/DEF2TZVP_FIELD_QMETRIC.csv` | Table S6 |
| Separate dicopper active-space check | `data/Cu2O2_def2SVP_parent/PARENT_CASCI2816_LOCAL_RESPONSE.csv` | Fig. S6, Table S7 |

The dicopper 25-point production scan is singlet orbital-relaxed
CASSCF(8,6)/def2-TZVP. The independent three-geometry parent-CI check is
fixed-orbital CASCI(28,16)/def2-SVP on separately optimized CASSCF(8,6)
source orbitals. The latter is a local basis-specific check, not a larger
def2-TZVP calculation. The archive provides the O–O and Cu–Cu distances along
the path, but not full Cartesian coordinates of all relaxed structures.

The F2 data establish the primary isotropic maximum and the near-2.7 Å
transverse-metric enhancement. Outer-region quadratic guides in SI Fig. S4
are not certified additional stationary points.

## Verify and reproduce the derived-data analysis

The distributed archive uses Python 3 with NumPy and Matplotlib:

```bash
unzip response_data.zip -d response_data
cd response_data
bash verify_manifest.sh
bash run_reproduction.sh
```

The first command checks the SHA-256 hashes of distributed inputs. The second
checks the derived numerical results and regenerates analysis figures into
`generated/`. The original electronic-structure wavefunction checkpoints,
integral files, and scratch data are not distributed, so the archive does not
recompute every first-principles wavefunction from the beginning.
