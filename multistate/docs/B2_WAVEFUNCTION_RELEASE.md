# B2 wavefunction release and audit (JCP multistate study)

The source SA3-CASSCF wavefunctions currently reside on the research server and **are not yet deposited on GitHub or Zenodo**. This guide describes the verified reconstruction results and how to prepare a separate Zenodo archive without altering source files.

## Completed audit (17 November 2026)

- The v1p8 production batch contributes 126 finite-field checkpoints at 7 bond lengths, while v1p9 contributes 108 at 6 additional bond lengths.
- 234 unique SHA256 hashes, 117 unique physical task keys with both signs, and 13 geometries; no duplicate or missing pair keys. All 234 `result.csv` rows show `PASS` for that production validation.
- Three archived SA3 CI vectors per field checkpoint are normalized and mutually orthogonal to numerical precision.
- Using the original production `frame_overlap` implementation and archived MO/CI vectors, all 234 zero-to-field overlap matrices were reconstructed. Maximum Frobenius error versus stored values: **1.1433814963738287e-15**. Maximum singular-value error: **9.992007221626409e-16**. Maximum transport Frobenius error: **1.851111708424076e-15**.
- 117 finite-field plus/minus overlaps were recalculated. This **does not** establish a separate independent-algorithm verification or a complete reproduction of the paper's hybrid response observables.
- Original `AMBIGUOUS` flags from other comparisons must not be overwritten by the fact that these 234 stage-2c tasks pass.

## GitHub versus Zenodo

**GitHub** is for scripts, config templates, manifest / pair mapping CSVs, audit reports, figure scripts, and derived analysis data. **Zenodo** is for the raw HDF5 checkpoints and source-task records (plus matching scripts/configs and checksums).

The raw `*.h5` files are not in GitHub and must not be described as publicly deposited until the upload is complete. The full dataset should be assigned its Zenodo DOI only after the record has been deposited and verified.

## On the server

From `/scr/zdb/B2_quantum_geometry/B2_polar_qgeom_package`:

```bash
python3 prepare_B2_zenodo.py --out B2_release
cd B2_release
sha256sum B2_SA3_wavefunctions_v1.tar.gz
tar -tzf B2_SA3_wavefunctions_v1.tar.gz | tail
```

The script [`scripts/prepare_B2_zenodo.py`](../scripts/prepare_B2_zenodo.py) validates the 234 signed-field tasks and all 13 zero-field checkpoint counts, creates `B2_SA3_wavefunctions_v1.tar.gz`, and emits `SHA256SUMS`, `file_inventory.csv`, `pair_mapping.csv`, and `release_report.json`.

For extra rigor, extract the archive to a temporary directory and run `sha256sum -c SHA256SUMS` inside it. Consider separate upstream state-specific (stage 1b) and stage 2b source archives after mapping them to paper claims; do not claim that the 247 main SA3 files alone recreate the whole hybrid model.

## Release checks

1. Confirm all 234 signed-field and 13 zero-field checkpoints are present and all manifest/checkpoint mapping checks pass.
2. Preserve original CSV statuses, scientific limitations, software versions, and the CASSCF source configuration.
3. Check the created archive SHA256, uncompressed manifest and extraction test.
4. Upload the archive to Zenodo; save its DOI, record version, files, and checksum.
5. Only then update main-text DAS and SI S8 to describe the **specific** newly available B2 wavefunction materials. C2 and HO2 limitations remain separate.

No production wavefunctions have been copied to GitHub as part of this repository update.
