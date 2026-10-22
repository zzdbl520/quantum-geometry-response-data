# HO2 quantum-geometry protocol validation v1.2.5

This package decides whether the two robust bond-parallel polarizability
maxima found in the original equal-weight SA2-CASSCF(13,9)/aug-cc-pVTZ pilot
are physical ground-state response features or artifacts of changing the
second state in the SA2 orbital-optimization objective.

The original pilot work directory is read-only.  All new checkpoints are
written below a separate validation work directory.

## What is calculated

1. **SS zero transport:** state-specific ground-state CASSCF(13,9) over source
   0--20, using the original SA2 seed only as an initial orbital/CI guess.
2. **SA4 zero transport:** equal-weight four-state CASSCF(13,9) over the same
   grid.  The exact nonorthogonal many-electron overlap tests whether a smooth
   four-state manifold absorbs the SA2 two-state replacements at source 7 and
   18.
3. **SS finite fields:** four magnitudes and both signs at source 4--6, 9--11,
   and 15--18 (80 tasks).
4. **Robust topology:** a maximum must pass a conservative fit-uncertainty
   envelope, have `Xi_left > 0` and `Xi_right < 0`, have negative local
   log-curvature with a vertex inside its bracket, and occur in at least 75%
   of 12 raw-field/extrapolation schemes.

The CAS is unchanged: two O(1s) cores are frozen and the complete
O(2s,2p)+O(2s,2p)+H(1s) valence space gives CAS(13,9), with
5 A1 + 2 B1 + 2 B2 active orbitals in exact linear C2v.

## Requirements

- Python 3.11
- PySCF
- NumPy, SciPy, h5py, Matplotlib
- A completed original v1.0.1 pilot work directory containing
  `zero/irc_013.h5` and preferably `analysis/qgeom_summary.csv`

Run the dependency-light package check first:

```bash
python self_test.py
```

## Recommended launch

From the directory containing the unpacked package:

```bash
package="$PWD/HO2_qgeom_protocol_validation_v1"
pilot_workdir="/scr/zdb/HO2_qgeom_pilot_work"
validation_workdir="/scr/zdb/HO2_qgeom_protocol_validation_work"

nohup env WORKERS=4 THREADS_PER_WORKER=4 MEMORY_MB=30000 \
  bash "$package/run_protocol_validation.sh" \
  "$pilot_workdir" "$validation_workdir" \
  > "$validation_workdir.pipeline.stdout" 2>&1 &

tail -f "$validation_workdir.pipeline.stdout"
```

`MEMORY_MB` is per worker.  Four 30-GB field workers require at least 120 GB
plus headroom.  SS and SA4 zero transports are sequential because each one is
a directly continued chain.

## Staged launch

### 1. Prepare an isolated 0--20 grid

```bash
python "$package/prepare_protocol.py" \
  --pilot-workdir "$pilot_workdir" \
  --workdir "$validation_workdir" \
  --indices 0:20
```

### 2. Run SS and SA4 zero transport

```bash
nohup python -u "$package/run_protocol_zero.py" \
  --workdir "$validation_workdir" \
  --protocol both \
  --threads 4 \
  --memory-mb 30000 \
  > "$validation_workdir/zero_protocols.stdout" 2>&1 &

tail -f "$validation_workdir/zero_protocols.stdout"
```

The command is restartable.  Converged files are skipped.  A low adjacent
state/subspace overlap is recorded as a scientific diagnostic rather than
silently changing the CAS.

### 3. Prepare and run the 80 SS field tasks

```bash
python "$package/prepare_ss_field_tasks.py" \
  --workdir "$validation_workdir" \
  --indices "4:6,9:11,15:18" \
  --fields "2.5e-4,5.0e-4,7.5e-4,1.0e-3"

manifest="$validation_workdir/protocol_validation/ss/field_manifest.jsonl"

nohup python -u "$package/run_ss_field_tasks.py" \
  --manifest "$manifest" \
  --workdir "$validation_workdir" \
  --workers 4 \
  --threads-per-worker 4 \
  --memory-mb 30000 \
  > "$validation_workdir/ss_field_tasks.stdout" 2>&1 &

tail -f "$validation_workdir/ss_field_tasks.stdout"
```

Individual task IDs and inclusive slices can be rescued with `--task-ids`,
for example `--task-ids 8` or `--task-ids 16:23`.

### 4. Analyze and plot

```bash
python "$package/analyze_protocol_validation.py" \
  --workdir "$validation_workdir" \
  --memory-mb 30000 \
  | tee "$validation_workdir/protocol_analysis.stdout"

python "$package/plot_protocol_validation.py" \
  --workdir "$validation_workdir"
```

The legacy single-root protocol diagnostic is:

```text
protocol_validation/analysis/protocol_validation_gate.json
```

It is retained for reproducibility.  Its fixed source-5/10/16 peak gate and
full-rank SA4 continuity gate are intentionally conservative; failure does
not reject a response-relevant continuous subspace at a crossing.

## Gauge-invariant rank-2 manifold response

Version 1.1 adds a local SA4 signed-field calculation in which the two-state
response manifold is selected from four field-state candidates by exact
many-electron subspace overlap.  Field frames are polar-transported into the
zero-field gauge before differentiating the effective Hamiltonian.  The
result separates:

- the covariant polarizability operator;
- the non-Abelian projector quantum metric;
- the internal dipole operator, whose spectral width diagnoses first-order
  response within the manifold; and
- the effective response-energy operator obtained by congruence.

Run the default rank-2 calculation over the two mechanism/peak windows
(sources 2--11 and 15--18) with:

```bash
package="$PWD/HO2_qgeom_protocol_validation_v1"
validation_workdir="/scr/zdb/HO2_qgeom_protocol_validation_work"

nohup env WORKERS=4 THREADS_PER_WORKER=4 MEMORY_MB=30000 \
  bash "$package/run_manifold_response.sh" "$validation_workdir" \
  > "$validation_workdir/manifold_response.stdout" 2>&1 &

tail -f "$validation_workdir/manifold_response.stdout"
```

Version 1.2.5 uses the common production window
`2.5e-4,5.0e-4,7.5e-4,1.0e-3` a.u. and separates two decisions:

- `operator_extrapolation_gate` tests every transported matrix element;
- `observable_mechanism_gate` tests invariant raw-field plateau envelopes;
- `local_extrema_validation_gate` tests only bracketed peak/nonpeak decisions;
- `reaction_topology_gate` combines those local decisions with stable rank.

The reconstructed lowest-state response is separated into the external term
`A_00` and the retained-manifold term `2|M_01|^2/(E_1-E_0)`.  A robust maximum
requires its lower four-field envelope to exceed both neighboring upper
envelopes.  Sources 5 and 9 are the validated maxima; sources 8, 10, 16, and
17 are explicit nonmaximum controls in `target_extremum_status`.  A separate
global plateau warning may remain at endpoint 18 without changing these local
topology decisions.

The new decisive output is:

```text
protocol_validation/analysis/manifold/manifold_response_gate.json
```

The calculation never overwrites the SS fields or the SA4 zero checkpoints.

Possible peak labels are:

- `ROBUST_MAXIMUM`
- `ROBUST_NOT_MAXIMUM`
- `CANDIDATE_MAXIMUM`
- `NOT_MAXIMUM`
- `UNBRACKETED`

`protocol_validation_gate` is the earlier root-specific SS/SA4 control and is
retained as an audit.  The physical rank-2 conclusion is read from
`manifold_response_gate.json`: `reaction_topology_gate` validates the local
two-maximum topology, whereas `observable_plateau_gate` remains the stricter
all-source control.  They are intentionally reported separately.

## Tail extension after the local manifold and SS maps pass

The bundled reaction-1 dataset already contains source 21--57.  No new IRC or
geometry interpolation is required.  Extend only after the 0--20 protocol
test passes:

```bash
python "$package/prepare_protocol.py" \
  --pilot-workdir "$pilot_workdir" \
  --workdir "$validation_workdir" \
  --indices 0:30

python -u "$package/run_protocol_zero.py" \
  --workdir "$validation_workdir" \
  --protocol both \
  --direction upper \
  --stop-source-index 30 \
  --threads 4 \
  --memory-mb 30000

python "$package/prepare_ss_field_tasks.py" \
  --workdir "$validation_workdir" \
  --indices "4:6,9:11,15:30" \
  --fields "2.5e-4,5.0e-4,7.5e-4,1.0e-3"

python -u "$package/run_ss_field_tasks.py" \
  --manifest "$validation_workdir/protocol_validation/ss/field_manifest.jsonl" \
  --workdir "$validation_workdir" \
  --workers 4 \
  --threads-per-worker 4 \
  --memory-mb 30000
```

Existing 0--20 zero checkpoints and completed local-window field tasks are
skipped.  Continue to source 40 only if the response has not reached an
asymptote or bracketed another extremum by source 30.

## Main outputs

```text
protocol_validation/ss/zero/irc_*.h5
protocol_validation/sa4/zero/irc_*.h5
protocol_validation/ss/fields/irc_*/F_*.h5
protocol_validation/analysis/ss_qgeom_pair_data.csv
protocol_validation/analysis/ss_qgeom_summary.csv
protocol_validation/analysis/topology_sensitivity.csv
protocol_validation/analysis/zero_protocol_audit.csv
protocol_validation/analysis/protocol_validation_gate.json
protocol_validation/analysis/HO2_protocol_validation.pdf
protocol_validation/sa4/manifold_fields/irc_*/F_*.h5
protocol_validation/analysis/manifold/manifold_pair_data.csv
protocol_validation/analysis/manifold/manifold_summary.csv
protocol_validation/analysis/manifold/manifold_response_gate.json
protocol_validation/analysis/manifold/HO2_manifold_response.pdf
protocol_validation/analysis/manifold/HO2_response_mechanisms.pdf
protocol_validation/analysis/manifold/HO2_response_mechanisms.png
protocol_validation/analysis/manifold/HO2_SI_late_path_control.pdf
protocol_validation/analysis/manifold/HO2_SI_late_path_control.png
```

## Interpretation

- If source 5 and 16 remain robust SS maxima and the SA4 four-state manifold
  is continuous, the two mechanisms can be interpreted as physical ground-
  state response reversals rather than SA2 protocol artifacts.
- Source 10 becomes a third maximum only if it passes the uncertainty envelope
  and topology-support gates.  Otherwise it is an incipient stationary
  cancellation or shoulder.
- If SS and SA2 disagree strongly near source 16--18, do not average the two
  results.  The disagreement is evidence of variational-protocol sensitivity
  and must be resolved before extending the scientific claim.
