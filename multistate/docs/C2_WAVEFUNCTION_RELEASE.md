# C2 SA6 wavefunction archive preparation

The main C2 production data were located on gu03 under `/mnt/C2_quantum_geometry/C2_SA4_pilot`. The directory name is historical (SA4 pilot), but the relevant source calculations are **SA6-CASSCF(8,8)**.

## Current preliminary inventory (author-supplied audit, 17 November 2026)

- `C2_SA6_global_transport_results_v1/points/`: **119** zero-field SA6 HDF5 files, one per geometry.
- `C2_SA6_global_field_work_v2/fields/`: **888** signed-field SA6 HDF5 files spanning **54** geometries.
- **444** unique `(R, axis, |F|)` keys and 444 positive/negative candidate pairs. Original HDF5 sign attribute is numerical `-1` or `+1`, not `minus` or `plus`. The initially reported zero complete pairs was a normalization error in the first audit script.
- All 1007 HDF5 checkpoint files inspected were readable, showed expected `(6,70,70)` CI, `(92,92)` MO and six-state energy arrays, carried a `converged=True` flag, and had unique SHA256 hashes. This does **not** establish all root-continuity or response gates.
- Additional separately scoped controls: 27 local transport HDF5 files, 96 crossing-field HDF5 files, and nine HDF5 files each in the two z-closure folders. These should not be treated as duplicate global-production tasks.

## Release preparation

Use [`scripts/prepare_C2_zenodo.py`](../scripts/prepare_C2_zenodo.py) on gu03:

```bash
cd /mnt/C2_quantum_geometry/C2_SA4_pilot
python3 prepare_C2_zenodo.py --out C2_release
cd C2_release
sha256sum C2_SA6_wavefunctions_v1.tar.gz
mkdir -p verify_extract
tar -xzf C2_SA6_wavefunctions_v1.tar.gz -C verify_extract
(cd verify_extract && sha256sum -c SHA256SUMS)
```

The script refuses to generate the archive if checkpoint counts, signed-field pairs, zero-field geometry coverage or required HDF5 records fail checks. It does not modify raw checkpoint data. Run with `--include-controls` to add historic local/crossing/z-closure control files, with their counts reported separately.

The resultant compressed archive must be uploaded to Zenodo manually, and its DOI added here **after** it is published and verified. GitHub holds scripts and audit reports; the raw HDF5 wavefunctions belong on Zenodo.

This is presently an archive and structural consistency audit, **not** a full independent wavefunction-overlap reproduction analogous to the completed B2 audit. The final DAS/SI S8 should describe only work actually performed.
