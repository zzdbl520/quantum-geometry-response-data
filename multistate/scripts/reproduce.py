#!/usr/bin/env python3
"""Verify the distributed records and regenerate current figures."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

R = Path(__file__).resolve().parents[1]
env = dict(os.environ, MPLBACKEND='Agg')

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=R,env=env,check=True)

run(R/'scripts/verify_revision.py')
run(R/'scripts/verify_molecular_tables.py')
run(R/'scripts/make_figures.py')
c = R/'evidence/C2'
run(R/'scripts/plot_c2_current.py',
    '--zero-root',c/'C2_SA6_global_transport_results_v1',
    '--field-work',c/'C2_SA6_global_field_work_v2',
    '--diagnostic-roots',c/'C2_SA6_z_closure_R_2p9',
    c/'C2_SA6_z_closure_diagnostic_v1',c/'C2_SA6_z_closure_R_3p2',
    '--output-prefix',R/'figures/C2','--dpi','200')
v = json.loads((R/'verification/verification_report.json').read_text())
def num(x):
    a,b=f'{x:.2e}'.split('e')
    return f'${a}\\times10^{{{int(b)}}}$'
entries = [
 ('Positive-gap factorization',v['positive_support']['congruence_relative_error']),
 ('Signed-gap factorization',v['signed_support']['congruence_relative_error']),
 ('Largest unitary-covariance error',max(v[k]['covariance_relative_error'] for k in ['positive_support','signed_support'])),
 ('Sylvester residual (absolute)',v['split_cluster']['sylvester_residual']),
 ('Best finite-field metric error',v['split_cluster']['best_metric_relative_error']),
 ('Best aligned quadratic-matrix error',v['split_cluster']['best_A_relative_error']),
 ('Unsupported part of the split-cluster example',v['split_support_counterexample']['unsupported_relative_norm'])]
table = r'''\begin{table}[ht]
\caption{Finite-dimensional checks rerun for this release. Errors are relative
Frobenius errors unless indicated otherwise. The metric entry compares scalar
traces. Values can vary slightly with the numerical libraries.}
\begin{ruledtabular}\begin{tabular}{lr}
Check & Error\\
'''+''.join(f'{a} & {num(b)}'+r'\\'+'\n' for a,b in entries)+r'''
\end{tabular}\end{ruledtabular}
\end{table}
'''
(R/'supporting_information/verification_table.tex').write_text(table)
environment = {'python':platform.python_version(),
               **{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']}}
(R/'verification/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
print('PASS: theoretical checks, molecular table checks, and all current figure scripts.')
