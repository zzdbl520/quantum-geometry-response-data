# C2 certified multistate response figure

This package contains the publication figure, its exact plotting script,
the numerical audit, provenance, and paste-ready LaTeX caption and Results
text.

## Scientific policy

- Zero-field PES data come from the 119-point transport audit, which is PASS.
- Relaxed response curves contain only axis-specific PASS rows.
- Ground-state response curves are broken at failed rows and at changes of the zero-field ground-state identity.
- Panel-c quantities use component-specific certification: only gates acting on the plotted rank-three target response, target metric, signed-field pairs, and target common-frame intercept determine whether a point is shown.
- At 1.55 Angstrom the rank-three component audit is PASS even though the aggregate audit is FAIL because an auxiliary rank-six response-matrix stability measure fails. The point is therefore retained in panel c, with the distinction recorded in both provenance and audit CSV.
- The purple guides are internal X/Delta crossings: both states belong to the rank-three target projector, so its geometric curves remain connected.
- The gray guide at 1.524121 Angstrom is an external Delta/Pi target-environment crossing. The metric is shown as two one-sided branches because the target projector loses spectral isolation, whereas the geometry-weighted scale changes sign continuously through zero.
- No finite quadratic-response value is assigned at the exact external closure; the red segment records the one-sided limiting trend.
- The frozen-orbital CASCI calculations diagnose continuity only; their response values are never spliced into relaxed-CASSCF curves.
- The optimized longitudinal failures from 2.9 to 3.2 Angstrom are displayed as diagnostic singular values rather than reported as physical response data.
- In panel d, the nearly coincident fixed-CASCI star and open-diamond series are displaced horizontally by -0.004 and +0.004 Angstrom solely for visibility; the underlying bond lengths are unchanged.

## Reproduce

From the directory containing the original analysis roots:

```bash
python plot_c2_certified_final.py \
  --zero-root C2_SA6_global_transport_results_v1 \
  --field-work C2_SA6_global_field_work_v2 \
  --diagnostic-roots \
    C2_SA6_z_closure_R_2p9 \
    C2_SA6_z_closure_diagnostic_v1 \
    C2_SA6_z_closure_R_3p2 \
  --output-prefix C2_certified_multistate_response
```

Expected outputs are PDF, PNG, SVG, and a provenance JSON file.
