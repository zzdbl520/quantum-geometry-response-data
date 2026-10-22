# B2 certified cusp-response figure

Publication-ready figure generated from the completed Stage 2C response package and the Stage 1B symmetry-resolved PES map.

## Certification

- 63/63 paired signed-field rows passed.
- 21/21 axis-response limits passed.
- 7/7 near-crossing geometries passed.
- The exact crossing is plotted as a true cusp, with no line or finite response value inserted at \(R_c\).
- Direct and relaxed transported transition moments remain separate; their ratio is diagnostic only.

## Files

- `B2_certified_cusp_response.pdf`: vector publication figure.
- `B2_certified_cusp_response.svg`: editable vector figure.
- `B2_certified_cusp_response.png`: 600 dpi preview.
- `plot_B2_certified_cusp_response.py`: reproducible plotting and validation script.
- `B2_certified_cusp_response_provenance.json`: source paths, numerical values, and plotting policy.
- `B2_certified_cusp_response_audit.csv`: compact certification audit.
- `B2_figure_caption.tex`: concise manuscript caption.

## Reproduction

```bash
python plot_B2_certified_cusp_response.py \
  --stage2c-dir /path/to/B2_stage2c_runs \
  --branch-map /path/to/B2_symmetry_branch_map.csv \
  --output-prefix B2_certified_cusp_response
```
