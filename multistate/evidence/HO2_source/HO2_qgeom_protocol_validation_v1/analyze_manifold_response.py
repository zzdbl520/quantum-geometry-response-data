#!/usr/bin/env python3
"""Analyze gauge-invariant rank-m response from signed-field SA4 frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ho2_core import (
    build_molecule,
    grid_by_source_index,
    parse_floats,
    read_csv,
    write_csv,
    write_json,
)
from manifold_core import (
    aligned_hamiltonian,
    covariant_pair_operators,
    effective_energy_operator,
    fit_even_matrix,
    ground_state_response_decomposition,
    plus_minus_metric_trace,
    quantum_metric_operator,
)
from protocol_core import (
    SUBSPACE_OVERLAP_WARN,
    VERSION,
    complete_h5,
    load_h5,
    load_metadata,
    same_geometry_overlap_n,
    validation_root,
    zero_path,
)
from run_manifold_field_task import complete_manifold_field, read_manifest


LOW_GAP_EH = 2.5e-2
NUMERICAL_TOLERANCE = 5.0e-3
FIT_STABILITY_TOLERANCE = 5.0e-3
METRIC_CLOSURE_TOLERANCE = 5.0e-2
DEFAULT_PLATEAU_FIELDS = (2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3)
OBSERVABLE_PLATEAU_TOLERANCE = 3.0e-2


PLATEAU_OBSERVABLES = {
    "external_ground_alpha_au": "external_ground_alpha",
    "internal_manifold_ground_alpha_au": "internal_manifold_ground_alpha",
    "reconstructed_ground_alpha_au": "reconstructed_ground_alpha",
    "trace_alpha_per_state_au": "trace_alpha_per_state",
    "trace_metric_per_state_au": "trace_metric_per_state",
}


def selected_frame(data, indices):
    index = np.asarray(indices, dtype=int)
    return {
        "mo_coeff": np.asarray(data["mo_coeff"]),
        "ci_vectors": np.asarray(data["ci_vectors"])[index],
    }


def stable_rank_from_zero(zero, threshold=SUBSPACE_OVERLAP_WARN):
    values = zero["attrs"].get("parent_subspace_sv")
    if values in (None, "null"):
        return None, []
    singular_values = [float(value) for value in values]
    return sum(value >= threshold for value in singular_values), singular_values


def incident_ss_overlap(workdir: Path, source_index: int, selected_sources):
    values = []
    for index in selected_sources:
        path = zero_path(workdir, "ss", index)
        if not complete_h5(path, "ss"):
            continue
        data = load_h5(path)
        parent = data["attrs"].get("parent_source_index")
        overlap = data["attrs"].get("parent_ground_best_overlap")
        if parent in (None, "null") or overlap in (None, "null"):
            continue
        if source_index in (index, int(parent)):
            values.append(float(overlap))
    return min(values) if values else float("nan")


def matrix_columns(prefix: str, matrix: np.ndarray):
    value = np.real(np.asarray(matrix))
    answer = {}
    for i in range(value.shape[0]):
        for j in range(i, value.shape[1]):
            answer[f"{prefix}_{i}{j}"] = float(value[i, j])
    return answer


def eig_columns(prefix: str, matrix: np.ndarray):
    values = np.linalg.eigvalsh(np.real(np.asarray(matrix)))
    return {
        f"{prefix}_{index}": float(value)
        for index, value in enumerate(values)
    }


def regularize_metric(metric: np.ndarray):
    value = 0.5 * (np.asarray(metric) + np.asarray(metric).T)
    eig, vec = np.linalg.eigh(value)
    negative = float(min(eig.min(), 0.0))
    eig = np.maximum(eig, 0.0)
    return (vec * eig) @ vec.T, negative


def select_plateau_pairs(local_pairs, requested_fields):
    """Return one raw pair for every requested field, within float tolerance."""
    selected = []
    missing = []
    for target in requested_fields:
        matches = [
            row for row in local_pairs
            if np.isclose(
                float(row["field_au"]), float(target), rtol=1.0e-10, atol=1.0e-14
            )
        ]
        if len(matches) == 1:
            selected.append(matches[0])
        else:
            missing.append(float(target))
    return selected, missing


def plateau_statistics(records, key):
    """Conservative median/envelope statistics for an invariant observable."""
    values = np.asarray([float(record[key]) for record in records], dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        return {
            "median": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "half_range": float("nan"),
            "relative_range": float("inf"),
        }
    median = float(np.median(values))
    lower = float(np.min(values))
    upper = float(np.max(values))
    return {
        "median": median,
        "lower": lower,
        "upper": upper,
        "half_range": 0.5 * (upper - lower),
        "relative_range": (upper - lower) / max(abs(median), 1.0),
    }


def plateau_columns(prefix, statistics):
    return {
        f"plateau_{prefix}_median_au": float(statistics["median"]),
        f"plateau_{prefix}_lower_au": float(statistics["lower"]),
        f"plateau_{prefix}_upper_au": float(statistics["upper"]),
        f"plateau_{prefix}_half_range_au": float(statistics["half_range"]),
        f"plateau_{prefix}_relative_range": float(statistics["relative_range"]),
    }


def mark_plateau_extrema(summary_rows):
    """Classify maxima by nonoverlapping raw-field plateau envelopes."""
    by_source = {int(row["source_index"]): row for row in summary_rows}
    extrema = []
    status = {}
    for index, row in sorted(by_source.items()):
        left = by_source.get(index - 1)
        right = by_source.get(index + 1)
        if left is None or right is None:
            status[index] = "UNBRACKETED"
            continue
        center = float(row["plateau_reconstructed_ground_alpha_median_au"])
        left_value = float(left["plateau_reconstructed_ground_alpha_median_au"])
        right_value = float(right["plateau_reconstructed_ground_alpha_median_au"])
        if not (center > left_value and center >= right_value):
            center_upper = float(
                row["plateau_reconstructed_ground_alpha_upper_au"]
            )
            neighbor_lower = max(
                float(left["plateau_reconstructed_ground_alpha_lower_au"]),
                float(right["plateau_reconstructed_ground_alpha_lower_au"]),
            )
            status[index] = (
                "ROBUST_NOT_MAXIMUM"
                if neighbor_lower > center_upper else "NOT_MAXIMUM"
            )
            continue
        lower = float(row["plateau_reconstructed_ground_alpha_lower_au"])
        neighbor_upper = max(
            float(left["plateau_reconstructed_ground_alpha_upper_au"]),
            float(right["plateau_reconstructed_ground_alpha_upper_au"]),
        )
        robust = bool(lower > neighbor_upper)
        status[index] = "ROBUST_MAXIMUM" if robust else "CANDIDATE_MAXIMUM"

        external = float(row["plateau_external_ground_alpha_median_au"])
        external_is_maximum = bool(
            external
            > float(left["plateau_external_ground_alpha_median_au"])
            and external
            >= float(right["plateau_external_ground_alpha_median_au"])
        )
        internal = float(
            row["plateau_internal_manifold_ground_alpha_median_au"]
        )
        total = center
        if robust and not external_is_maximum and internal > 0.0:
            mechanism = "internal-manifold-generated"
        elif float(row["internal_gap_zero_Eh"]) < LOW_GAP_EH:
            mechanism = "mixed-low-gap-manifold"
        else:
            mechanism = "smooth-external-manifold"
        extrema.append({
            "observable": "reconstructed-ground",
            "source_index": index,
            "s_cart_A": float(row["s_cart_A"]),
            "status": status[index],
            "median_au": center,
            "lower_envelope_au": lower,
            "upper_envelope_au": float(
                row["plateau_reconstructed_ground_alpha_upper_au"]
            ),
            "neighbor_upper_envelope_au": neighbor_upper,
            "conservative_peak_margin_au": lower - neighbor_upper,
            "mechanism_class": mechanism,
            "external_ground_is_local_maximum": external_is_maximum,
            "external_ground_alpha_median_au": external,
            "internal_manifold_ground_alpha_median_au": internal,
            "internal_response_fraction": abs(internal) / max(abs(total), 1.0e-14),
            "internal_gap_zero_Eh": float(row["internal_gap_zero_Eh"]),
        })
    return extrema, status


def robust_adjacent_orderings(summary_rows):
    """Report adjacent source pairs whose full field envelopes do not overlap."""
    by_source = {int(row["source_index"]): row for row in summary_rows}
    orderings = []
    for left_index in sorted(by_source):
        right_index = left_index + 1
        if right_index not in by_source:
            continue
        left = by_source[left_index]
        right = by_source[right_index]
        left_lower = float(
            left["plateau_reconstructed_ground_alpha_lower_au"]
        )
        left_upper = float(
            left["plateau_reconstructed_ground_alpha_upper_au"]
        )
        right_lower = float(
            right["plateau_reconstructed_ground_alpha_lower_au"]
        )
        right_upper = float(
            right["plateau_reconstructed_ground_alpha_upper_au"]
        )
        if left_upper < right_lower:
            direction = "INCREASING"
            margin = right_lower - left_upper
        elif right_upper < left_lower:
            direction = "DECREASING"
            margin = left_lower - right_upper
        else:
            direction = "OVERLAPPING"
            margin = -min(left_upper - right_lower, right_upper - left_lower)
        orderings.append({
            "left_source_index": left_index,
            "right_source_index": right_index,
            "direction": direction,
            "conservative_margin_au": float(margin),
            "left_envelope_au": [left_lower, left_upper],
            "right_envelope_au": [right_lower, right_upper],
        })
    return orderings


def mark_extrema(summary_rows, key: str, label: str):
    by_source = {int(row["source_index"]): row for row in summary_rows}
    extrema = []
    for index, row in sorted(by_source.items()):
        left = by_source.get(index - 1)
        right = by_source.get(index + 1)
        if left is None or right is None:
            continue
        center_value = float(row[key])
        left_value = float(left[key])
        right_value = float(right[key])
        if center_value > left_value and center_value >= right_value:
            uncertainty = float(row["alpha_matrix_fit_uncertainty_au"])
            robust = bool(
                center_value - uncertainty
                > max(
                    left_value + float(left["alpha_matrix_fit_uncertainty_au"]),
                    right_value + float(right["alpha_matrix_fit_uncertainty_au"]),
                )
            )
            internal = float(row["internal_manifold_ground_alpha_au"])
            total = float(row["reconstructed_ground_alpha_au"])
            if label == "reconstructed-ground" and np.isfinite(internal):
                internal_fraction = abs(internal) / max(abs(total), 1.0e-14)
                mechanism = (
                    "internal-manifold-dominated"
                    if internal_fraction >= 0.5
                    else "mixed-manifold-response"
                )
            else:
                internal_fraction = float("nan")
                mechanism = (
                    "low-gap-manifold"
                    if float(row["internal_gap_zero_Eh"]) < LOW_GAP_EH
                    else "smooth-external-manifold"
                )
            extrema.append({
                "observable": label,
                "source_index": index,
                "s_cart_A": float(row["s_cart_A"]),
                "value_au": center_value,
                "robust_envelope_gate": robust,
                "mechanism_class": mechanism,
                "internal_gap_zero_Eh": float(row["internal_gap_zero_Eh"]),
                "ss_local_minimum_overlap": float(row["ss_local_minimum_overlap"]),
                "external_ground_alpha_au": float(
                    row["external_ground_alpha_au"]
                ),
                "internal_manifold_ground_alpha_au": internal,
                "internal_response_fraction": internal_fraction,
            })
    return extrema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--memory-mb", type=int, default=30000)
    parser.add_argument(
        "--plateau-fields",
        default=",".join(f"{value:g}" for value in DEFAULT_PLATEAU_FIELDS),
        help="Common field magnitudes used for invariant envelope validation",
    )
    parser.add_argument(
        "--target-sources",
        default="5,8,9,10,16,17",
        help="Comma-separated candidate extrema to summarize",
    )
    args = parser.parse_args()

    plateau_fields = parse_floats(args.plateau_fields)
    if len(plateau_fields) < 3:
        raise SystemExit("--plateau-fields requires at least three magnitudes")
    target_sources = sorted({
        int(value) for value in args.target_sources.split(",") if value.strip()
    })

    workdir = Path(args.workdir).resolve()
    root = validation_root(workdir)
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest else root / "sa4" / "manifold_field_manifest.jsonl"
    )
    records = read_manifest(manifest)
    if not records:
        raise SystemExit(f"Empty manifold field manifest: {manifest}")
    rank_values = {int(record["manifold_rank"]) for record in records}
    if len(rank_values) != 1:
        raise SystemExit(f"Manifest mixes manifold ranks: {sorted(rank_values)}")
    rank = rank_values.pop()
    selected_sources = sorted({int(record["source_index"]) for record in records})
    rows = grid_by_source_index(workdir)
    metadata = load_metadata(workdir)

    tasks = {}
    missing_tasks = []
    for record in records:
        source_index = int(record["source_index"])
        field = abs(float(record["signed_field_au"]))
        sign = int(np.sign(float(record["signed_field_au"])))
        path = Path(record["output_path"])
        if not complete_manifold_field(path, rank):
            missing_tasks.append(int(record["task_id"]))
            continue
        tasks.setdefault((source_index, field), {})[sign] = path

    pilot_path = Path(metadata["pilot_workdir"]) / "analysis" / "qgeom_summary.csv"
    pilot = (
        {int(row["source_index"]): row for row in read_csv(pilot_path)}
        if pilot_path.exists() else {}
    )
    ss_path = root / "analysis" / "ss_qgeom_summary.csv"
    ss_summary = (
        {int(row["source_index"]): row for row in read_csv(ss_path)}
        if ss_path.exists() else {}
    )

    pair_rows = []
    summary_rows = []
    missing_pairs = []
    for source_index in selected_sources:
        zero_file = zero_path(workdir, "sa4", source_index)
        if not complete_h5(zero_file, "sa4"):
            missing_pairs.append({"source_index": source_index, "reason": "sa4_zero"})
            continue
        zero = load_h5(zero_file)
        zero_energy = np.asarray(zero["e_states_Eh"], dtype=float)[:rank]
        h_zero = np.diag(zero_energy)
        row = rows[source_index]
        mol = build_molecule(
            row,
            basis=str(zero["attrs"]["basis"]),
            memory_mb=args.memory_mb,
            symmetry=True,
        )
        fields = sorted(field for index, field in tasks if index == source_index)
        valid_fields = []
        dipole_matrices = []
        alpha_matrices = []
        metric_matrices = []
        metric_pm_traces = []
        local_pairs = []
        for field in fields:
            signed = tasks.get((source_index, field), {})
            if -1 not in signed or 1 not in signed:
                missing_pairs.append({
                    "source_index": source_index,
                    "field_au": field,
                    "reason": "signed_pair",
                })
                continue
            minus = load_h5(signed[-1])
            plus = load_h5(signed[1])
            minus_indices = np.asarray(minus["selected_manifold_indices"], dtype=int)
            plus_indices = np.asarray(plus["selected_manifold_indices"], dtype=int)
            zero_minus = np.asarray(minus["zero_state_overlap"])[
                :rank, minus_indices
            ]
            zero_plus = np.asarray(plus["zero_state_overlap"])[
                :rank, plus_indices
            ]
            minus_frame = selected_frame(minus, minus_indices)
            plus_frame = selected_frame(plus, plus_indices)
            minus_plus = same_geometry_overlap_n(minus_frame, plus_frame, mol)
            h_minus, sv_minus = aligned_hamiltonian(
                zero_minus, np.asarray(minus["e_states_Eh"])[minus_indices]
            )
            h_plus, sv_plus = aligned_hamiltonian(
                zero_plus, np.asarray(plus["e_states_Eh"])[plus_indices]
            )
            dipole, alpha = covariant_pair_operators(
                h_zero, h_minus, h_plus, field
            )
            metric = quantum_metric_operator(zero_minus, zero_plus, field)
            metric_pm_trace = plus_minus_metric_trace(minus_plus, field)
            alpha_eig = np.linalg.eigvalsh(alpha)
            metric_eig = np.linalg.eigvalsh(metric)
            dipole_eig = np.linalg.eigvalsh(dipole)
            metric_trace = float(np.trace(metric).real)
            pair_ground_response = ground_state_response_decomposition(
                zero_energy, dipole, alpha
            )
            pair_internal = float(
                pair_ground_response["internal_ground_alpha"]
            )
            pair_reconstructed = float(
                pair_ground_response["reconstructed_ground_alpha"]
            )
            record = {
                "source_index": source_index,
                "s_cart_A": float(row["s_cart_A"]),
                "field_au": field,
                "manifold_rank": rank,
                "minus_selected_indices": ";".join(map(str, minus_indices)),
                "plus_selected_indices": ";".join(map(str, plus_indices)),
                "minus_anchor_sv_min": float(np.min(sv_minus)),
                "plus_anchor_sv_min": float(np.min(sv_plus)),
                "minus_plus_sv_min": float(np.min(
                    np.linalg.svd(minus_plus, compute_uv=False)
                )),
                "trace_alpha_per_state_au": float(np.trace(alpha).real / rank),
                "principal_alpha_min_au": float(alpha_eig.min()),
                "principal_alpha_max_au": float(alpha_eig.max()),
                "trace_metric_per_state_au": metric_trace / rank,
                "plus_minus_metric_per_state_au": metric_pm_trace / rank,
                "metric_estimator_relative_difference": abs(
                    metric_trace - metric_pm_trace
                ) / max(abs(metric_trace), abs(metric_pm_trace), 1.0e-14),
                "trace_Delta_geom_Eh": (
                    float(np.trace(alpha).real / (2.0 * metric_trace))
                    if metric_trace > 0.0 else float("nan")
                ),
                "internal_dipole_width_au": float(dipole_eig.max() - dipole_eig.min()),
                "external_ground_alpha_au": float(
                    pair_ground_response["external_ground_alpha"]
                ),
                "internal_manifold_ground_alpha_au": pair_internal,
                "reconstructed_ground_alpha_au": pair_reconstructed,
                "internal_response_fraction": (
                    abs(pair_internal) / max(abs(pair_reconstructed), 1.0e-14)
                ),
                "perturbative_internal_response_supported": int(
                    pair_ground_response[
                        "perturbative_internal_response_supported"
                    ]
                ),
                "maximum_abs_spin_square_error": max(
                    abs(float(value) - 0.75)
                    for data, indices in (
                        (zero, np.arange(rank)),
                        (minus, minus_indices),
                        (plus, plus_indices),
                    )
                    for value in np.asarray(data["state_spin_square"])[indices]
                ),
            }
            record.update(matrix_columns("dipole", dipole))
            record.update(matrix_columns("alpha", alpha))
            record.update(matrix_columns("metric", metric))
            pair_rows.append(record)
            local_pairs.append(record)
            valid_fields.append(field)
            dipole_matrices.append(dipole)
            alpha_matrices.append(alpha)
            metric_matrices.append(metric)
            metric_pm_traces.append(metric_pm_trace)

        if len(valid_fields) < 3:
            missing_pairs.append({
                "source_index": source_index,
                "reason": f"only_{len(valid_fields)}_field_magnitudes",
            })
            continue
        plateau_pairs, missing_plateau_fields = select_plateau_pairs(
            local_pairs, plateau_fields
        )
        plateau_stats = {
            prefix: plateau_statistics(plateau_pairs, key)
            for key, prefix in PLATEAU_OBSERVABLES.items()
        }
        plateau_failures = []
        if missing_plateau_fields:
            plateau_failures.append("MISSING_PLATEAU_FIELDS")
        for prefix, statistics in plateau_stats.items():
            if statistics["relative_range"] > OBSERVABLE_PLATEAU_TOLERANCE:
                plateau_failures.append(f"{prefix.upper()}_RANGE")
        plateau_metric_closure = (
            max(
                float(pair["metric_estimator_relative_difference"])
                for pair in plateau_pairs
            )
            if plateau_pairs else float("inf")
        )
        if plateau_metric_closure >= METRIC_CLOSURE_TOLERANCE:
            plateau_failures.append("PLATEAU_METRIC_CLOSURE")
        dipole0, dipole_residual, dipole_stability = fit_even_matrix(
            valid_fields, dipole_matrices
        )
        alpha0, alpha_residual, alpha_stability = fit_even_matrix(
            valid_fields, alpha_matrices
        )
        metric0_raw, metric_residual, metric_stability = fit_even_matrix(
            valid_fields, metric_matrices
        )
        metric0, metric_negative = regularize_metric(metric0_raw)
        metric_pm0 = np.polynomial.polynomial.polyfit(
            np.asarray(valid_fields) ** 2, metric_pm_traces, 2
        )[0]
        metric_trace = float(np.trace(metric0).real)
        metric_closure = abs(metric_trace - metric_pm0) / max(
            abs(metric_trace), abs(metric_pm0), 1.0e-14
        )
        delta0, delta_info = effective_energy_operator(alpha0, metric0)
        ground_response = ground_state_response_decomposition(
            zero_energy, dipole0, alpha0
        )
        alpha_eig = np.linalg.eigvalsh(alpha0)
        metric_eig = np.linalg.eigvalsh(metric0)
        dipole_eig = np.linalg.eigvalsh(dipole0)
        delta_eig = np.linalg.eigvalsh(delta0)
        stable_rank, parent_sv = stable_rank_from_zero(zero)
        ss_overlap = incident_ss_overlap(workdir, source_index, range(21))
        alpha_uncertainty = max(alpha_residual, alpha_stability) * max(
            float(np.max(np.abs(alpha_eig))), 1.0
        )
        minimum_anchor_sv = min(
            min(pair["minus_anchor_sv_min"], pair["plus_anchor_sv_min"])
            for pair in local_pairs
        )
        maximum_spin_error = max(
            pair["maximum_abs_spin_square_error"] for pair in local_pairs
        )
        numerical_failures = []
        for name, value, threshold in (
            ("ALPHA_FIT_RESIDUAL", alpha_residual, NUMERICAL_TOLERANCE),
            ("ALPHA_FIT_STABILITY", alpha_stability, FIT_STABILITY_TOLERANCE),
            ("DIPOLE_FIT_RESIDUAL", dipole_residual, NUMERICAL_TOLERANCE),
            ("DIPOLE_FIT_STABILITY", dipole_stability, FIT_STABILITY_TOLERANCE),
            ("METRIC_FIT_RESIDUAL", metric_residual, NUMERICAL_TOLERANCE),
            ("METRIC_FIT_STABILITY", metric_stability, FIT_STABILITY_TOLERANCE),
            ("METRIC_CLOSURE", metric_closure, METRIC_CLOSURE_TOLERANCE),
        ):
            if value >= threshold:
                numerical_failures.append(name)
        if minimum_anchor_sv < SUBSPACE_OVERLAP_WARN:
            numerical_failures.append("FIELD_SUBSPACE")
            plateau_failures.append("FIELD_SUBSPACE")
        if maximum_spin_error >= 1.0e-5:
            numerical_failures.append("SPIN")
            plateau_failures.append("SPIN")
        if not ground_response["perturbative_internal_response_supported"]:
            numerical_failures.append("DEGENERATE_INTERNAL_RESPONSE")
            plateau_failures.append("DEGENERATE_INTERNAL_RESPONSE")
        numerical_gate = not numerical_failures
        observable_plateau_gate = not plateau_failures
        pilot_row = pilot.get(source_index, {})
        ss_row = ss_summary.get(source_index, {})
        classification = (
            "LOW_GAP_MANIFOLD_REGION"
            if float(zero_energy[1] - zero_energy[0]) < LOW_GAP_EH
            or (np.isfinite(ss_overlap) and ss_overlap < SUBSPACE_OVERLAP_WARN)
            else "SMOOTH_MANIFOLD_REGION"
        )
        summary = {
            "source_index": source_index,
            "grid_position": int(row["grid_position"]),
            "s_cart_A": float(row["s_cart_A"]),
            "manifold_rank": rank,
            "trace_energy_zero_Eh": float(np.mean(zero_energy)),
            "internal_gap_zero_Eh": float(zero_energy[1] - zero_energy[0]),
            "trace_alpha_per_state_au": float(np.trace(alpha0).real / rank),
            "principal_alpha_min_au": float(alpha_eig.min()),
            "principal_alpha_max_au": float(alpha_eig.max()),
            "trace_metric_per_state_au": metric_trace / rank,
            "trace_Delta_geom_Eh": (
                float(np.trace(alpha0).real / (2.0 * metric_trace))
                if metric_trace > 0.0 else float("nan")
            ),
            "internal_dipole_width_au": float(dipole_eig.max() - dipole_eig.min()),
            "external_ground_alpha_au": float(
                ground_response["external_ground_alpha"]
            ),
            "internal_manifold_ground_alpha_au": float(
                ground_response["internal_ground_alpha"]
            ),
            "reconstructed_ground_alpha_au": float(
                ground_response["reconstructed_ground_alpha"]
            ),
            "internal_response_fraction": (
                abs(float(ground_response["internal_ground_alpha"]))
                / max(
                    abs(float(ground_response["reconstructed_ground_alpha"])),
                    1.0e-14,
                )
            ),
            "perturbative_internal_response_supported": int(
                ground_response["perturbative_internal_response_supported"]
            ),
            "plateau_field_count": len(plateau_pairs),
            "plateau_field_magnitudes_au": ";".join(
                f"{float(pair['field_au']):.8g}" for pair in plateau_pairs
            ),
            "plateau_metric_closure_maximum": float(plateau_metric_closure),
            "observable_plateau_gate": int(observable_plateau_gate),
            "observable_plateau_failure_reasons": ";".join(plateau_failures),
            "effective_Delta_min_Eh": float(delta_eig.min()),
            "effective_Delta_max_Eh": float(delta_eig.max()),
            "metric_supported_rank": int(delta_info["metric_supported_rank"]),
            "metric_condition_number": float(delta_info["metric_condition_number"]),
            "metric_fit_minimum_raw_eigenvalue": metric_negative,
            "metric_estimator_closure_relative": float(metric_closure),
            "alpha_matrix_fit_relative_residual": alpha_residual,
            "alpha_matrix_fit_relative_stability": alpha_stability,
            "alpha_matrix_fit_uncertainty_au": alpha_uncertainty,
            "metric_matrix_fit_relative_residual": metric_residual,
            "metric_matrix_fit_relative_stability": metric_stability,
            "dipole_matrix_fit_relative_residual": dipole_residual,
            "dipole_matrix_fit_relative_stability": dipole_stability,
            "minimum_field_anchor_sv": minimum_anchor_sv,
            "minimum_minus_plus_sv": min(
                pair["minus_plus_sv_min"] for pair in local_pairs
            ),
            "maximum_abs_spin_square_error": maximum_spin_error,
            "geometry_parent_stable_rank": (
                stable_rank if stable_rank is not None else rank
            ),
            "geometry_parent_sv": ";".join(f"{value:.8f}" for value in parent_sv),
            "ss_local_minimum_overlap": ss_overlap,
            "sa2_ground_alpha_au": float(
                pilot_row.get("alpha_parallel_au", "nan")
            ),
            "ss_ground_alpha_au": float(
                ss_row.get("alpha_parallel_au", "nan")
            ),
            "region_classification": classification,
            "numerical_gate": int(numerical_gate),
            "numerical_failure_reasons": ";".join(numerical_failures),
        }
        for prefix, statistics in plateau_stats.items():
            summary.update(plateau_columns(prefix, statistics))
        plateau_internal = float(
            summary["plateau_internal_manifold_ground_alpha_median_au"]
        )
        plateau_total = float(
            summary["plateau_reconstructed_ground_alpha_median_au"]
        )
        summary["plateau_internal_response_fraction"] = (
            abs(plateau_internal) / max(abs(plateau_total), 1.0e-14)
        )
        summary.update(matrix_columns("dipole0", dipole0))
        summary.update(matrix_columns("alpha0", alpha0))
        summary.update(matrix_columns("metric0", metric0))
        summary.update(matrix_columns("Delta0", delta0))
        summary.update(eig_columns("principal_alpha", alpha0))
        summary.update(eig_columns("metric_eigenvalue", metric0))
        summary.update(eig_columns("effective_Delta", delta0))
        summary_rows.append(summary)

    summary_rows.sort(key=lambda item: item["grid_position"])
    pair_rows.sort(key=lambda item: (item["source_index"], item["field_au"]))
    operator_extrema = (
        mark_extrema(summary_rows, "trace_alpha_per_state_au", "trace-average")
        + mark_extrema(summary_rows, "principal_alpha_max_au", "principal-maximum")
        + mark_extrema(
            summary_rows,
            "reconstructed_ground_alpha_au",
            "reconstructed-ground",
        )
    )
    operator_extrapolation_gate = bool(
        not missing_tasks
        and not missing_pairs
        and len(summary_rows) == len(selected_sources)
        and all(bool(row["numerical_gate"]) for row in summary_rows)
    )
    observable_plateau_gate = bool(
        not missing_tasks
        and not missing_pairs
        and len(summary_rows) == len(selected_sources)
        and all(bool(row["observable_plateau_gate"]) for row in summary_rows)
    )
    stable_rank_gate = bool(
        summary_rows
        and all(int(row["geometry_parent_stable_rank"]) >= rank for row in summary_rows)
    )
    observable_extrema, extremum_status_by_source = mark_plateau_extrema(
        summary_rows
    )
    target_extremum_status = {
        str(index): extremum_status_by_source.get(index, "NOT_COMPUTED")
        for index in target_sources
    }
    robust_terminal_statuses = {"ROBUST_MAXIMUM", "ROBUST_NOT_MAXIMUM"}
    local_extrema_validation_gate = bool(
        observable_extrema
        and all(
            target_extremum_status[str(index)] in robust_terminal_statuses
            for index in target_sources
        )
    )
    robust_orderings = robust_adjacent_orderings(summary_rows)
    reaction_topology_gate = bool(
        stable_rank_gate and local_extrema_validation_gate
    )
    observable_mechanism_gate = bool(
        observable_plateau_gate and stable_rank_gate
    )
    numerical_failures = [
        {
            "source_index": int(row["source_index"]),
            "reasons": [
                value for value in str(row["numerical_failure_reasons"]).split(";")
                if value
            ],
            "alpha_fit_residual": float(
                row["alpha_matrix_fit_relative_residual"]
            ),
            "alpha_fit_stability": float(
                row["alpha_matrix_fit_relative_stability"]
            ),
            "dipole_fit_residual": float(
                row["dipole_matrix_fit_relative_residual"]
            ),
            "dipole_fit_stability": float(
                row["dipole_matrix_fit_relative_stability"]
            ),
        }
        for row in summary_rows if not bool(row["numerical_gate"])
    ]
    failure_names = sorted({
        reason
        for item in numerical_failures
        for reason in item["reasons"]
    })
    observable_plateau_failures = [
        {
            "source_index": int(row["source_index"]),
            "reasons": [
                value
                for value in str(
                    row["observable_plateau_failure_reasons"]
                ).split(";")
                if value
            ],
            "reconstructed_relative_range": float(
                row["plateau_reconstructed_ground_alpha_relative_range"]
            ),
            "trace_alpha_relative_range": float(
                row["plateau_trace_alpha_per_state_relative_range"]
            ),
            "metric_relative_range": float(
                row["plateau_trace_metric_per_state_relative_range"]
            ),
        }
        for row in summary_rows if not bool(row["observable_plateau_gate"])
    ]
    plateau_failure_names = sorted({
        reason
        for item in observable_plateau_failures
        for reason in item["reasons"]
    })
    report = {
        "package_version": VERSION,
        "manifold_rank": rank,
        "candidate_state_count": 4,
        "selected_source_indices": selected_sources,
        "missing_task_ids": sorted(set(missing_tasks)),
        "missing_pairs": missing_pairs,
        "numerical_gate": operator_extrapolation_gate,
        "operator_extrapolation_gate": operator_extrapolation_gate,
        "observable_plateau_gate": observable_plateau_gate,
        "observable_mechanism_gate": observable_mechanism_gate,
        "local_extrema_validation_gate": local_extrema_validation_gate,
        "reaction_topology_gate": reaction_topology_gate,
        "response_relevant_stable_rank_gate": stable_rank_gate,
        "manifold_response_gate": observable_mechanism_gate,
        "plateau_field_magnitudes_au": plateau_fields,
        "observable_plateau_relative_range_tolerance": (
            OBSERVABLE_PLATEAU_TOLERANCE
        ),
        "extrema": observable_extrema,
        "operator_extrema": operator_extrema,
        "target_extremum_status": target_extremum_status,
        "validated_extremum_sources": [
            int(item["source_index"])
            for item in observable_extrema
            if item["status"] == "ROBUST_MAXIMUM"
        ],
        "robust_adjacent_orderings": robust_orderings,
        "extrema_status": (
            "VALIDATED_OBSERVABLE_PLATEAUS"
            if observable_mechanism_gate
            else (
                "VALIDATED_LOCAL_EXTREMA_WITH_GLOBAL_CONTROL_WARNING"
                if reaction_topology_gate
                else "PROVISIONAL_LOCAL_TOPOLOGY_GATE_FAILED"
            )
        ),
        "numerical_failures": numerical_failures,
        "numerical_failure_counts": {
            name: sum(name in item["reasons"] for item in numerical_failures)
            for name in failure_names
        },
        "observable_plateau_failures": observable_plateau_failures,
        "observable_plateau_failure_counts": {
            name: sum(
                name in item["reasons"] for item in observable_plateau_failures
            )
            for name in plateau_failure_names
        },
        "classification_counts": {
            name: sum(row["region_classification"] == name for row in summary_rows)
            for name in ("LOW_GAP_MANIFOLD_REGION", "SMOOTH_MANIFOLD_REGION")
        },
        "interpretation": (
            "The covariant response operator, non-Abelian projector metric, "
            "and internal dipole operator are transported into the zero-field "
            "rank-m gauge. The physical nondegenerate lowest-state response "
            "is reconstructed as A_00 plus its retained-manifold mixing term. "
            "Invariant raw-field plateau envelopes validate physical extrema "
            "separately from the stricter full-matrix extrapolation gate. "
            "Local nonoverlap can validate extrema and monotonic orderings even "
            "when an unrelated boundary control exceeds the global plateau "
            "range tolerance. "
            "A lost fourth SA4 root does not fail a continuous rank-2 response "
            "manifold."
        ),
    }
    analysis = root / "analysis" / "manifold"
    write_csv(analysis / "manifold_pair_data.csv", pair_rows)
    write_csv(analysis / "manifold_summary.csv", summary_rows)
    write_json(analysis / "manifold_response_gate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if not missing_tasks and not missing_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
