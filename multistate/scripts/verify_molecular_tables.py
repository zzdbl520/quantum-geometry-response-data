#!/usr/bin/env python3
"""Recompute manuscript quantities from distributed tables, without QC jobs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'
OUT = ROOT / 'verification'
OUT.mkdir(exist_ok=True)

def close(a, b, rtol=1e-9, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def crossings(r, gap):
    found = []
    for i in range(len(r)-1):
        if gap[i] * gap[i+1] < 0:
            found.append(float(r[i] - gap[i]*(r[i+1]-r[i])/(gap[i+1]-gap[i])))
    return found

c = E/'C2/C2_SA6_global_transport_results_v1/analysis'
p = pd.read_csv(c/'transport_points.csv').sort_values('R_angstrom')
edges = pd.read_csv(c/'transport_edges.csv')
summary = json.loads((c/'transport_summary.json').read_text())
assert len(p) == 119 and len(edges) == 118
internal = crossings(p.R_angstrom.to_numpy(), p.X_minus_Delta_gap_Eh.to_numpy())
external = crossings(p.R_angstrom.to_numpy(), (p.E_Delta_mean_Eh-p.E_Pi_mean_Eh).to_numpy())
close(internal, [x['linear_crossing_estimate_angstrom'] for x in summary['crossing_brackets']])
close(external, [1.5241207081393648], rtol=1e-8)
mins = {}
for column, key in [('active_min_sv','minimum_active_subspace_sv'),
                    ('retained_min_sv','minimum_retained_subspace_sv'),
                    ('state_projector_min_sv','minimum_state_projector_sv'),
                    ('minimum_matched_abs_state_overlap','minimum_matched_abs_state_overlap')]:
    mins[key] = float(edges[column].min())
    close(mins[key], summary[key])
mu = float(p.max_abs_Pdeg_mu_Pdeg_au.max())
close(mu, summary['maximum_abs_Pdeg_mu_Pdeg_au'], atol=1e-25)
assert mu < 1e-12

b = E/'B2/B2_stage2_combined_runs/stage2c_runs'
final = pd.read_csv(b/'B2_final_near_crossing_response.csv')
pairs = pd.read_csv(b/'transported_field_pairs.csv')
hybrid = pd.read_csv(b/'hybrid_pair_audit.csv')
states = pd.read_csv(b/'field_states.csv')
axes = pd.read_csv(b/'hybrid_axis_response.csv')
assert (len(final),len(pairs),len(states),len(axes)) == (13,117,234,39)
assert not pairs.duplicated(['R_Angstrom','axis','field_magnitude_au']).any()
close(final.mu_relaxed_transport_perpendicular_au/final.mu_direct_perpendicular_au,
      final.orbital_response_renormalization_ratio)
assert (hybrid.original_status.astype(str) == pairs.status.astype(str)).all()
assert hybrid.diagnostic_flags.fillna('').str.contains('direct_mu_control').any()
cross = final.loc[np.abs(final.stage1b_X_minus_A_gap_Eh).idxmin()]
assert abs(cross.stage1b_X_minus_A_gap_Eh) < 1e-8
close(cross.cusp_slope_direct_perpendicular_au, cross.mu_direct_perpendicular_au)
# At the cusp the table must not silently report a finite transverse value.
assert pd.isna(pd.to_numeric(cross.alpha_x_final_au, errors='coerce'))
assert pd.isna(pd.to_numeric(cross.metric_x_final_au, errors='coerce'))

h = pd.read_csv(E/'HO2/manifold_pair_data.csv')
hs = pd.read_csv(E/'HO2/manifold_summary.csv').set_index('source_index')
hg = json.loads((E/'HO2/manifold_response_gate.json').read_text())
assert len(h) == 56 and h.source_index.nunique() == 14
assert set(h.source_index) == set(range(2,12)) | set(range(15,19))
assert not h.duplicated(['source_index','field_au']).any()
assert (h.groupby('source_index').size() == 4).all()
close(sorted(h.field_au.unique()), [.00025,.0005,.00075,.001])
gap = h.source_index.map(hs.internal_gap_zero_Eh)
internal_response = 2 * np.abs(h.dipole_01)**2/gap
close(internal_response, h.internal_manifold_ground_alpha_au)
close(h.alpha_00, h.external_ground_alpha_au)
total = h.external_ground_alpha_au + internal_response
close(total, h.reconstructed_ground_alpha_au)
peak_rows = []
for i in [5,9]:
    central = h[h.source_index == i]
    neighbors = h[h.source_index.isin([i-1,i+1])]
    vals = central.reconstructed_ground_alpha_au
    row = {'source_index':i,'median':float(vals.median()),'lower':float(vals.min()),
           'upper':float(vals.max()),
           'margin':float(vals.min()-neighbors.reconstructed_ground_alpha_au.max())}
    archived = next(x for x in hg['extrema'] if x['source_index'] == i)
    close(row['median'],archived['median_au'])
    close(row['margin'],archived['conservative_peak_margin_au'])
    assert row['margin'] > 0
    peak_rows.append(row)
assert len(hg['numerical_failures']) == 12
assert not hg['operator_extrapolation_gate'] and not hg['observable_plateau_gate']
geom = E/'HO2_source/HO2_qgeom_protocol_validation_v1/data/01_irc.npz'
metadata = json.loads((E/'HO2/protocol_metadata.json').read_text())
assert hashlib.sha256(geom.read_bytes()).hexdigest() == metadata['source_sha256']

report = {
    'status':'PASS',
    'scope':'Checks of distributed records; no new wavefunctions or overlap reconstruction.',
    'C2':{'points':len(p),'edges':len(edges),'internal_crossings_A':internal,
          'external_crossing_A':external,'minimum_singular_values':mins,'max_projected_dipole_au':mu},
    'B2':{'geometries':len(final),'signed_states':len(states),'pairs':len(pairs),'axis_rows':len(axes),
          'crossing_A':float(cross.R_Angstrom),'direct_coupling_au':float(cross.mu_direct_perpendicular_au),
          'aligned_derivative_magnitude_au':float(cross.mu_relaxed_transport_perpendicular_au),
          'ratio_range':[float(final.orbital_response_renormalization_ratio.min()),
                         float(final.orbital_response_renormalization_ratio.max())],
          'original_pair_status_counts':pairs.status.value_counts().to_dict(),
          'hybrid_pair_status_counts':hybrid.status.value_counts().to_dict()},
    'HO2':{'pairs':len(h),'geometries':h.source_index.nunique(),
           'internal_formula_max_error':float(np.max(np.abs(internal_response-h.internal_manifold_ground_alpha_au))),
           'min_field_anchor_sv':float(h[['minus_anchor_sv_min','plus_anchor_sv_min']].min().min()),
           'peak_checks':peak_rows,'full_matrix_failure_count':len(hg['numerical_failures']),
           'operator_extrapolation_gate':hg['operator_extrapolation_gate'],
           'global_scalar_range_gate':hg['observable_plateau_gate']}
}
(OUT/'molecular_table_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
