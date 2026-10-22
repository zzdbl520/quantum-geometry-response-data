#!/usr/bin/env python3
"""Analyze SS response fields and SS/SA4 zero-field protocol continuity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ho2_core import (
    build_molecule,
    fit_even_limit,
    load_grid,
    read_csv,
    write_csv,
    write_json,
)
from protocol_core import (
    VERSION,
    complete_h5,
    load_h5,
    load_metadata,
    same_geometry_overlap_n,
    validation_root,
    zero_path,
)
from run_ss_field_task import read_manifest


HARTREE_TO_EV = 27.211386245988
TARGET_PEAKS = (5, 10, 16)

EXTREMUM_DEFAULTS = {
    "xi_left": float("nan"),
    "xi_right": float("nan"),
    "topology_support_count": 0,
    "topology_scheme_count": 0,
    "topology_support_fraction": float("nan"),
    "robust_envelope_gate": 0,
    "local_vertex_bracket_gate": 0,
    "extremum_status": "UNBRACKETED",
    "local_vertex_s_A": float("nan"),
    "d2logg_local_ds2": float("nan"),
    "d2logDelta_local_ds2": float("nan"),
    "d2logalpha_local_ds2": float("nan"),
    "mechanism_class": "unassigned",
}


def ensure_extremum_schema(rows) -> None:
    """Give bracketed and unbracketed summary rows one stable CSV schema."""
    for row in rows:
        for key, value in EXTREMUM_DEFAULTS.items():
            row.setdefault(key, value)


def mechanism_label(curvature_g: float, curvature_delta: float) -> str:
    negative_g = max(-float(curvature_g), 0.0)
    negative_delta = max(-float(curvature_delta), 0.0)
    if negative_g > 1.25 * max(negative_delta, 1.0e-14):
        return "metric-curvature-dominated"
    if negative_delta > 1.25 * max(negative_g, 1.0e-14):
        return "energy-scale-curvature-dominated"
    return "mixed-curvature"


def poly_limit(fields, values, selection, degree):
    f = np.asarray(fields, dtype=float)[selection]
    y = np.asarray(values, dtype=float)[selection]
    coeff = np.polynomial.polynomial.polyfit(f**2, y, min(degree, len(f) - 1))
    return float(coeff[0])


def fit_family(pair_by_source, summary_by_source):
    common = None
    for values in pair_by_source.values():
        fields = set(values)
        common = fields if common is None else common & fields
    fields = np.asarray(sorted(common or []), dtype=float)
    if len(fields) != 4:
        return fields, {}, []
    selections = {
        "reported_q2_all4": (np.arange(4), 2),
        "linear_all4": (np.arange(4), 1),
        "quadratic_low3": (np.asarray([0, 1, 2]), 2),
        "quadratic_high3": (np.asarray([1, 2, 3]), 2),
        "linear_low3": (np.asarray([0, 1, 2]), 1),
        "linear_high3": (np.asarray([1, 2, 3]), 1),
        "linear_low2": (np.asarray([0, 1]), 1),
        "linear_high2": (np.asarray([2, 3]), 1),
    }
    curves = {
        "reported_q2_all4": {
            index: float(row["alpha_parallel_au"])
            for index, row in summary_by_source.items()
        }
    }
    for name, (selection, degree) in selections.items():
        if name == "reported_q2_all4":
            continue
        curves[name] = {}
        for index, values in pair_by_source.items():
            y = [float(values[field]["alpha_parallel_au"]) for field in fields]
            curves[name][index] = poly_limit(fields, y, selection, degree)
    for field in fields:
        curves[f"raw_F={field:.2e}"] = {
            index: float(values[field]["alpha_parallel_au"])
            for index, values in pair_by_source.items()
        }
    topology_rows = []
    for scheme, curve in curves.items():
        for index, value in sorted(curve.items()):
            is_maximum = bool(
                index - 1 in curve
                and index + 1 in curve
                and value > curve[index - 1]
                and value >= curve[index + 1]
            )
            topology_rows.append({
                "scheme": scheme,
                "source_index": index,
                "alpha_parallel_au": value,
                "is_local_maximum": int(is_maximum),
            })
    return fields, curves, topology_rows


def zero_audit(workdir: Path, rows):
    audit_rows = []
    summary = {}
    for protocol in ("ss", "sa4"):
        for row in rows:
            source_index = int(row["source_index"])
            path = zero_path(workdir, protocol, source_index)
            if not complete_h5(path, protocol):
                audit_rows.append({
                    "protocol": protocol,
                    "source_index": source_index,
                    "status": "MISSING",
                    "parent_subspace_sv_min": float("nan"),
                    "parent_ground_best_overlap": float("nan"),
                    "sa2_ground_to_protocol_ground_abs": float("nan"),
                    "retained_projection_sv_min": float("nan"),
                })
                continue
            data = load_h5(path)
            attrs = data["attrs"]
            parent_sv = attrs.get("parent_subspace_sv")
            if parent_sv == "null":
                parent_sv = None
            projection = attrs.get("projection")
            status = attrs.get("parent_continuity_status")
            if status in (None, "null"):
                status = "SEED"
            parent_ground_value = attrs.get("parent_ground_best_overlap")
            if parent_ground_value == "null":
                parent_ground_value = None
            sa2_ground_value = attrs.get("sa2_ground_to_protocol_ground_abs")
            if sa2_ground_value == "null":
                sa2_ground_value = None
            audit_rows.append({
                "protocol": protocol,
                "source_index": source_index,
                "status": status,
                "parent_subspace_sv_min": (
                    float(min(parent_sv)) if parent_sv is not None else float("nan")
                ),
                "parent_ground_best_overlap": (
                    float(parent_ground_value)
                    if parent_ground_value is not None
                    else float("nan")
                ),
                "sa2_ground_to_protocol_ground_abs": (
                    float(sa2_ground_value)
                    if sa2_ground_value is not None
                    else float("nan")
                ),
                "retained_projection_sv_min": (
                    float(projection["retained_projection_sv_min"])
                    if isinstance(projection, dict)
                    else float("nan")
                ),
            })

        subset = [item for item in audit_rows if item["protocol"] == protocol]
        complete = [item for item in subset if item["status"] != "MISSING"]
        parent = [item for item in complete if item["status"] != "SEED"]
        parent_sv = [item["parent_subspace_sv_min"] for item in parent]
        parent_ground = [item["parent_ground_best_overlap"] for item in parent]
        sa2_ground = [
            item["sa2_ground_to_protocol_ground_abs"]
            for item in complete
            if np.isfinite(item["sa2_ground_to_protocol_ground_abs"])
        ]
        if protocol == "ss":
            continuity_gate = bool(parent_ground and min(parent_ground) > 0.75)
        else:
            continuity_gate = bool(parent_sv and min(parent_sv) > 0.75)
        summary[protocol] = {
            "completed_count": len(complete),
            "expected_count": len(rows),
            "minimum_parent_subspace_sv": (
                min(parent_sv) if parent_sv else None
            ),
            "minimum_parent_ground_best_overlap": (
                min(parent_ground) if parent_ground else None
            ),
            "minimum_sa2_ground_to_protocol_ground_overlap": (
                min(sa2_ground) if sa2_ground else None
            ),
            "continuity_warning_indices": [
                item["source_index"]
                for item in parent
                if item["status"] != "PASS"
            ],
            "continuity_gate": bool(
                len(complete) == len(rows) and continuity_gate
            ),
        }
    return audit_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--memory-mb", type=int, default=30000)
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    metadata = load_metadata(workdir)
    rows = load_grid(workdir)
    row_by_source = {int(row["source_index"]): row for row in rows}
    root = validation_root(workdir)
    manifest_path = root / "ss" / "field_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit("Missing SS field manifest; run prepare_ss_field_tasks.py")
    manifest = read_manifest(manifest_path)

    tasks = {}
    missing_tasks = []
    for record in manifest:
        path = Path(record["output_path"])
        if not complete_h5(path, "ss"):
            missing_tasks.append(int(record["task_id"]))
            continue
        key = (int(record["source_index"]), abs(float(record["signed_field_au"])))
        tasks.setdefault(key, {})[int(np.sign(float(record["signed_field_au"])))]=path

    selected_sources = sorted({int(record["source_index"]) for record in manifest})
    pair_rows = []
    pair_by_source = {}
    summary_rows = []
    missing_pairs = []
    pilot_summary_path = Path(metadata["pilot_workdir"]) / "analysis" / "qgeom_summary.csv"
    pilot_summary = {}
    if pilot_summary_path.exists():
        pilot_summary = {
            int(row["source_index"]): row for row in read_csv(pilot_summary_path)
        }

    for source_index in selected_sources:
        row = row_by_source[source_index]
        zero_file = zero_path(workdir, "ss", source_index)
        if not complete_h5(zero_file, "ss"):
            missing_pairs.append({"source_index": source_index, "reason": "ss_zero"})
            continue
        zero = load_h5(zero_file)
        zero_energy = float(zero["e_states_Eh"][0])
        fields = sorted(magnitude for idx, magnitude in tasks if idx == source_index)
        valid_fields = []
        raw_alpha = []
        raw_metric = []
        raw_delta = []
        pair_subset = []
        mol = build_molecule(
            row,
            basis=str(zero["attrs"]["basis"]),
            memory_mb=args.memory_mb,
            symmetry=True,
        )
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
            overlap_pm = same_geometry_overlap_n(minus, plus, mol)
            fidelity = float(abs(overlap_pm[0, 0]) ** 2)
            alpha = -(
                float(plus["e_states_Eh"][0])
                + float(minus["e_states_Eh"][0])
                - 2.0 * zero_energy
            ) / field**2
            metric = (1.0 - fidelity) / (4.0 * field**2)
            delta = alpha / (2.0 * metric) if metric > 0.0 else float("nan")
            record = {
                "source_index": source_index,
                "s_cart_A": float(row["s_cart_A"]),
                "field_au": field,
                "alpha_parallel_au": alpha,
                "g_FF_au": metric,
                "Delta_geom_Eh": delta,
                "plus_minus_ground_fidelity": fidelity,
                "minus_zero_ground_overlap_abs": float(
                    minus["attrs"]["zero_ground_overlap_abs"]
                ),
                "plus_zero_ground_overlap_abs": float(
                    plus["attrs"]["zero_ground_overlap_abs"]
                ),
                "spin_square_zero": float(zero["state_spin_square"][0]),
                "spin_square_minus": float(minus["state_spin_square"][0]),
                "spin_square_plus": float(plus["state_spin_square"][0]),
            }
            pair_rows.append(record)
            pair_subset.append(record)
            valid_fields.append(field)
            raw_alpha.append(alpha)
            raw_metric.append(metric)
            raw_delta.append(delta)
        pair_by_source[source_index] = {
            float(record["field_au"]): record for record in pair_subset
        }
        if len(valid_fields) < 3:
            missing_pairs.append({
                "source_index": source_index,
                "reason": f"only_{len(valid_fields)}_field_magnitudes",
            })
            continue
        alpha0, alpha_residual, alpha_stability = fit_even_limit(
            valid_fields, raw_alpha
        )
        metric0, metric_residual, metric_stability = fit_even_limit(
            valid_fields, raw_metric
        )
        delta0 = alpha0 / (2.0 * metric0) if metric0 > 0.0 else float("nan")
        finite_delta = np.isfinite(raw_delta)
        if int(np.count_nonzero(finite_delta)) >= 3:
            delta_fit, delta_residual, delta_stability = fit_even_limit(
                np.asarray(valid_fields)[finite_delta],
                np.asarray(raw_delta)[finite_delta],
            )
        else:
            delta_fit = delta_residual = delta_stability = float("nan")
        alpha_uncertainty = max(alpha_residual, alpha_stability) * max(abs(alpha0), 1.0)
        pilot = pilot_summary.get(source_index)
        pilot_alpha = float(pilot["alpha_parallel_au"]) if pilot else float("nan")
        pilot_metric = float(pilot["g_FF_au"]) if pilot else float("nan")
        summary_rows.append({
            "source_index": source_index,
            "grid_position": int(row["grid_position"]),
            "s_cart_A": float(row["s_cart_A"]),
            "energy_zero_Eh": zero_energy,
            "alpha_parallel_au": alpha0,
            "g_FF_au": metric0,
            "Delta_geom_Eh": delta0,
            "Delta_geom_direct_fit_Eh": delta_fit,
            "Delta_fit_closure_relative_error": abs(delta_fit - delta0) / max(abs(delta0), 1.0e-14),
            "alpha_fit_relative_residual": alpha_residual,
            "alpha_fit_relative_stability": alpha_stability,
            "alpha_fit_uncertainty_au": alpha_uncertainty,
            "metric_fit_relative_residual": metric_residual,
            "metric_fit_relative_stability": metric_stability,
            "delta_fit_relative_residual": delta_residual,
            "delta_fit_relative_stability": delta_stability,
            "minimum_zero_ground_overlap_abs": min(
                min(item["minus_zero_ground_overlap_abs"], item["plus_zero_ground_overlap_abs"])
                for item in pair_subset
            ),
            "maximum_abs_spin_square_error": max(
                abs(item[key] - 0.75)
                for item in pair_subset
                for key in ("spin_square_zero", "spin_square_minus", "spin_square_plus")
            ),
            "sa2_alpha_parallel_au": pilot_alpha,
            "ss_minus_sa2_alpha_relative": (
                (alpha0 - pilot_alpha) / max(abs(pilot_alpha), 1.0e-14)
                if np.isfinite(pilot_alpha) else float("nan")
            ),
            "sa2_g_FF_au": pilot_metric,
            "ss_minus_sa2_metric_relative": (
                (metric0 - pilot_metric) / max(abs(pilot_metric), 1.0e-14)
                if np.isfinite(pilot_metric) else float("nan")
            ),
        })

    summary_rows.sort(key=lambda item: item["grid_position"])
    pair_rows.sort(key=lambda item: (item["source_index"], item["field_au"]))
    summary_by_source = {int(item["source_index"]): item for item in summary_rows}
    fields, curves, topology_rows = fit_family(pair_by_source, summary_by_source)
    scheme_count = len(curves)
    ensure_extremum_schema(summary_rows)

    extrema = []
    for index, item in summary_by_source.items():
        left = summary_by_source.get(index - 1)
        right = summary_by_source.get(index + 1)
        if left is None or right is None:
            continue
        alpha = float(item["alpha_parallel_au"])
        alpha_left = float(left["alpha_parallel_au"])
        alpha_right = float(right["alpha_parallel_au"])
        xi_left = np.log(alpha / alpha_left) / (
            float(item["s_cart_A"]) - float(left["s_cart_A"])
        )
        xi_right = np.log(alpha_right / alpha) / (
            float(right["s_cart_A"]) - float(item["s_cart_A"])
        )
        discrete = bool(alpha > alpha_left and alpha >= alpha_right)
        robust_envelope = bool(
            alpha - float(item["alpha_fit_uncertainty_au"])
            > max(
                alpha_left + float(left["alpha_fit_uncertainty_au"]),
                alpha_right + float(right["alpha_fit_uncertainty_au"]),
            )
        )
        support_count = 0
        for curve in curves.values():
            if index - 1 in curve and index in curve and index + 1 in curve:
                support_count += int(
                    curve[index] > curve[index - 1]
                    and curve[index] >= curve[index + 1]
                )
        support = support_count / scheme_count if scheme_count else float("nan")

        s3 = np.asarray([
            float(left["s_cart_A"]), float(item["s_cart_A"]), float(right["s_cart_A"])
        ])
        cg = np.polynomial.polynomial.polyfit(
            s3,
            np.log([
                float(left["g_FF_au"]), float(item["g_FF_au"]), float(right["g_FF_au"])
            ]),
            2,
        )
        cd = np.polynomial.polynomial.polyfit(
            s3,
            np.log([
                float(left["Delta_geom_Eh"]), float(item["Delta_geom_Eh"]), float(right["Delta_geom_Eh"])
            ]),
            2,
        )
        curvature_g = float(2.0 * cg[2])
        curvature_delta = float(2.0 * cd[2])
        curvature_total = curvature_g + curvature_delta
        csum = cg + cd
        vertex = float(-csum[1] / (2.0 * csum[2])) if csum[2] != 0.0 else float("nan")
        vertex_bracket_gate = bool(
            np.isfinite(vertex) and float(s3[0]) <= vertex <= float(s3[-1])
        )
        mechanism = mechanism_label(curvature_g, curvature_delta)
        if (
            discrete
            and robust_envelope
            and support >= 0.75
            and xi_left > 0.0
            and xi_right < 0.0
            and curvature_total < 0.0
            and vertex_bracket_gate
        ):
            status = "ROBUST_MAXIMUM"
        elif discrete and xi_left > 0.0 and xi_right < 0.0:
            status = "CANDIDATE_MAXIMUM"
        else:
            status = "NOT_MAXIMUM"
        item.update({
            "xi_left": float(xi_left),
            "xi_right": float(xi_right),
            "topology_support_count": support_count,
            "topology_scheme_count": scheme_count,
            "topology_support_fraction": float(support),
            "robust_envelope_gate": int(robust_envelope),
            "local_vertex_bracket_gate": int(vertex_bracket_gate),
            "extremum_status": status,
            "local_vertex_s_A": vertex,
            "d2logg_local_ds2": curvature_g,
            "d2logDelta_local_ds2": curvature_delta,
            "d2logalpha_local_ds2": curvature_total,
            "mechanism_class": mechanism,
        })
        if discrete:
            extrema.append({
                "source_index": index,
                "s_cart_A": float(item["s_cart_A"]),
                "alpha_parallel_au": alpha,
                "g_FF_au": float(item["g_FF_au"]),
                "Delta_geom_Eh": float(item["Delta_geom_Eh"]),
                "xi_left": float(xi_left),
                "xi_right": float(xi_right),
                "topology_support_fraction": float(support),
                "robust_envelope_gate": robust_envelope,
                "local_vertex_bracket_gate": vertex_bracket_gate,
                "extremum_status": status,
                "local_vertex_s_A": vertex,
                "mechanism_class": mechanism,
                "d2logg_local_ds2": curvature_g,
                "d2logDelta_local_ds2": curvature_delta,
                "d2logalpha_local_ds2": curvature_total,
            })

    analysis_dir = root / "analysis"
    write_csv(analysis_dir / "ss_qgeom_pair_data.csv", pair_rows)
    write_csv(analysis_dir / "ss_qgeom_summary.csv", summary_rows)
    write_csv(analysis_dir / "topology_sensitivity.csv", topology_rows)
    audit_rows, zero_summary = zero_audit(workdir, rows)
    write_csv(analysis_dir / "zero_protocol_audit.csv", audit_rows)

    numerical_gate = bool(
        not missing_tasks
        and not missing_pairs
        and len(summary_rows) == len(selected_sources)
        and len(fields) == 4
        and all(float(item["alpha_fit_relative_residual"]) < 5.0e-3 for item in summary_rows)
        and all(float(item["metric_fit_relative_residual"]) < 5.0e-3 for item in summary_rows)
        and all(float(item["minimum_zero_ground_overlap_abs"]) > 0.75 for item in summary_rows)
        and all(float(item["maximum_abs_spin_square_error"]) < 1.0e-5 for item in summary_rows)
    )
    robust_by_index = {
        int(item["source_index"]): item
        for item in extrema
        if item["extremum_status"] == "ROBUST_MAXIMUM"
    }
    two_peak_gate = 5 in robust_by_index and 16 in robust_by_index
    triple_peak_gate = two_peak_gate and 10 in robust_by_index
    distinct_mechanism_gate = bool(
        two_peak_gate
        and robust_by_index[5]["mechanism_class"]
        != robust_by_index[16]["mechanism_class"]
    )
    zero_gate = bool(
        zero_summary["ss"]["continuity_gate"]
        and zero_summary["sa4"]["continuity_gate"]
    )
    validation_gate = bool(
        numerical_gate and zero_gate and two_peak_gate and distinct_mechanism_gate
    )
    report = {
        "package_version": VERSION,
        "selected_field_source_indices": selected_sources,
        "field_magnitudes_au": fields.tolist(),
        "missing_task_ids": sorted(set(missing_tasks)),
        "missing_pairs": missing_pairs,
        "numerical_gate": numerical_gate,
        "zero_protocol_summary": zero_summary,
        "zero_protocol_gate": zero_gate,
        "extrema": extrema,
        "target_peak_status": {
            str(index): summary_by_source.get(index, {}).get("extremum_status", "NOT_COMPUTED")
            for index in TARGET_PEAKS
        },
        "two_robust_peak_gate": two_peak_gate,
        "triple_robust_peak_gate": triple_peak_gate,
        "distinct_mechanism_gate": distinct_mechanism_gate,
        "protocol_validation_gate": validation_gate,
        "decision": (
            "PHYSICAL_TWO_PEAKS_CONFIRMED"
            if validation_gate
            else "PROTOCOL_VALIDATION_INCOMPLETE_OR_FAILED"
        ),
        "interpretation_rule": (
            "A robust maximum must pass the conservative fit-uncertainty "
            "envelope, show Xi_left>0 and Xi_right<0, and be supported by at "
            "least 75% of raw-field/extrapolation schemes."
        ),
    }
    write_json(analysis_dir / "protocol_validation_gate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if not missing_tasks and not missing_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
