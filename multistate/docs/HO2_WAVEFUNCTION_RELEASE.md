# HO2 wavefunction archive: SA2 / SA4 / state-specific

The original HO2 calculations reside on cu02 under `/scr/zdb`. The three preserved protocols must remain distinct.

## Source archive inventory (17 November 2026)

| Protocol | Zero HDF5 | Signed-field HDF5 | Complete +/- pairs | Field IRC geometries |
| --- | ---: | ---: | ---: | ---: |
| SA2 pilot | 21 | 252 | 126 | 21 |
| SA4 manifold | 21 | 148 | 74 | 14 |
| State-specific | 21 | 136 | 68 | 17 |
| **Total** | **63** | **536** | **268** | |

The author-supplied structural audit read all **599** HDF5 wavefunction files, found **599 distinct SHA256 hashes**, and reported no missing/duplicate signed-field pairs, nonfinite CI/MO/energy arrays, or failed checkpoint convergence flags.

This is an **archive/structural audit**, not independent wavefunction-overlap reproduction or a clean-environment CASSCF rerun. In particular, the SA4 source has 74 pairs while the final paper's four-field manifold summary uses 56; and previously failed full-matrix extrapolation/global range tests must not be overwritten by simple checkpoint convergence flags.

## Prepare a Zenodo archive on cu02

Download [`prepare_HO2_zenodo.py`](../scripts/prepare_HO2_zenodo.py) to `/scr/zdb` and run:

```bash
python3 /scr/zdb/prepare_HO2_zenodo.py --root /scr/zdb --out /scr/zdb/HO2_release
cd /scr/zdb/HO2_release
mkdir -p verify_extract
tar -xzf HO2_wavefunctions_v1.tar.gz -C verify_extract
(cd verify_extract && sha256sum -c SHA256SUMS > ../verification.log 2>&1)
echo "Exit code: $?"
```

The package includes source checkpoint HDF5 files, original protocol and analysis metadata, Python source, SHA256 manifest, per-protocol pair mapping, and audit report. Review these materials before public release, and add the Zenodo DOI after posting and verifying the remote file. Zenodo is intended for the raw checkpoints; GitHub for the analytical records and scripts.

**No HO2 wavefunctions have been uploaded through this GitHub update.**
