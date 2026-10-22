#!/usr/bin/env python3
"""Extend a certified local C2 SA6 frame to a global bond-length window.

The local PASS archive is the immutable anchor.  New points are propagated
outward on the short- and long-bond sides, never by independent energy-sorted
optimizations.  The two sides are independent and may be run concurrently.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ANCHOR_R = 1.70
DEFAULT_R_MIN = 1.00
DEFAULT_R_MAX = 3.20
DEFAULT_REPORT_STEP = 0.05
DEFAULT_TRANSPORT_STEP = 0.025


def point_tag(r: float) -> str:
    return f"R_{r:.9f}".rstrip("0").rstrip(".").replace(".", "p")


def read_r(path: Path) -> float:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("This command requires h5py in the PySCF environment") from exc
    with h5py.File(path, "r") as h5:
        return float(h5.attrs["R_angstrom"])


def decimal_grid(start: float, stop: float, step: float) -> list[float]:
    start_d, stop_d, step_d = map(lambda x: Decimal(str(x)), (start, stop, step))
    if step_d <= 0 or stop_d < start_d:
        raise ValueError("Grid requires stop >= start and step > 0")
    count = int(((stop_d - start_d) / step_d).to_integral_value())
    values = [start_d + i * step_d for i in range(count + 1)]
    if values[-1] < stop_d - Decimal("1e-12"):
        values.append(stop_d)
    else:
        values[-1] = stop_d
    return [float(x) for x in values]


def certified_points(root: Path) -> dict[float, Path]:
    points = {}
    for h5_path in (root / "points").glob("R_*/point.h5"):
        summary_path = h5_path.parent / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "PASS":
                continue
        # The immutable center seed predates per-point summary.json files.  A
        # PASS local production_summary certifies it through the global
        # transport analysis, so it must not be dropped during import.
        points[read_r(h5_path)] = h5_path.parent
    return points


def import_local(local_root: Path, output: Path) -> dict[float, Path]:
    production = json.loads((local_root / "production_summary.json").read_text())
    if production.get("status") != "PASS":
        raise SystemExit("Local seed is not certified PASS")
    source = certified_points(local_root)
    if not source:
        raise SystemExit("No certified local point directories were found")
    destination_root = output / "points"
    destination_root.mkdir(parents=True, exist_ok=True)
    for r, source_dir in source.items():
        destination = destination_root / point_tag(r)
        if destination.exists():
            existing = destination / "point.h5"
            if not existing.exists() or abs(read_r(existing) - r) > 1.0e-9:
                raise RuntimeError(f"Existing destination conflicts at R={r:.12f}")
        else:
            shutil.copytree(source_dir, destination)
        destination_summary = destination / "summary.json"
        if not destination_summary.exists():
            destination_summary.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "R_angstrom": r,
                        "kind": "immutable_certified_local_center_seed",
                        "certification": (
                            "source local production_summary and rank-six "
                            "transport_summary are PASS"
                        ),
                        "source_point_directory": str(source_dir.resolve()),
                    },
                    indent=2,
                )
                + "\n"
            )
    (output / "local_seed_provenance.json").write_text(
        json.dumps(
            {
                "source": str(local_root.resolve()),
                "production_summary": production,
                "imported_point_count": len(source),
                "R_min_angstrom": min(source),
                "R_max_angstrom": max(source),
            },
            indent=2,
        )
        + "\n"
    )
    return source


def build_grid(
    local_r: list[float], r_min: float, r_max: float,
    report_step: float, transport_step: float,
) -> list[dict[str, object]]:
    local_min, local_max = min(local_r), max(local_r)
    if not r_min < local_min < local_max < r_max:
        raise ValueError("Requested global window must extend both sides of local seed")
    report = set(round(x, 10) for x in decimal_grid(r_min, r_max, report_step))
    short = decimal_grid(r_min, local_min, transport_step)
    long = decimal_grid(local_max, r_max, transport_step)
    all_r = sorted(set(round(x, 10) for x in short + long + local_r))
    local_set = set(round(x, 10) for x in local_r)
    rows = []
    for r in all_r:
        in_local = r in local_set
        is_report = r in report
        if in_local and is_report:
            role = "local_certified_report"
        elif in_local:
            role = "local_certified"
        elif is_report:
            role = "global_report"
        else:
            role = "transport_bridge"
        rows.append(
            {
                "R_angstrom": f"{r:.10f}",
                "side": "local" if in_local else ("short" if r < local_min else "long"),
                "grid_role": role,
                "plot_use": 1 if is_report else 0,
                "field_use": 1 if is_report else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = import_local(args.local_root.resolve(), output)
    rows = build_grid(
        list(source), args.R_min, args.R_max,
        args.report_step, args.transport_step,
    )
    write_csv(output / "geometry_grid.csv", rows)
    protocol = {
        "status": "READY",
        "method": "equal-weight symmetry-complete SA6-CASSCF(8,8)/aug-cc-pVTZ",
        "state_model": "3 Ag + B1g + B2u + B3u",
        "anchor_R_angstrom": ANCHOR_R,
        "R_window_angstrom": [args.R_min, args.R_max],
        "report_step_angstrom": args.report_step,
        "maximum_transport_step_angstrom": args.transport_step,
        "local_seed_point_count": len(source),
        "global_grid_point_count": len(rows),
        "field_point_count": sum(int(x["field_use"]) for x in rows),
        "ordering": "local PASS seed; adjacent outward continuation; polar-overlap identity transport",
        "response_lock": "finite fields remain locked until global transport_summary status is PASS",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


def grid_rows(output: Path) -> list[dict[str, str]]:
    with (output / "geometry_grid.csv").open() as handle:
        return list(csv.DictReader(handle))


def run_one(script_dir: Path, output: Path, r: float, restart: Path, memory: int) -> Path:
    point_dir = output / "points" / point_tag(r)
    point_h5 = point_dir / "point.h5"
    summary_path = point_dir / "summary.json"
    spin_penalty_shift = 0.0
    if point_h5.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "PASS":
            print(f"SKIP PASS {r:.9f} A", flush=True)
            return point_h5
        print(f"RESCUE REVIEW {r:.9f} A from its stored frame", flush=True)
        restart = point_h5
        failed_gates = {
            key for key, value in summary.get("gate", {}).items() if not value
        }
        if failed_gates == {"all_states_singlet_abs_S2_le_1e-6"}:
            spin_penalty_shift = 1.0
            print(
                "STRICT-SPIN RESCUE: add +1.0 Eh * S^2 penalty and report "
                "spin-penalty-corrected physical energies",
                flush=True,
            )
    point_dir.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{point_tag(r)}.stdout"
    command = [
        sys.executable, "-u", str(script_dir / "c2_sa6_transport_point.py"),
        "--R", f"{r:.12f}", "--restart-h5", str(restart),
        "--output", str(point_dir), "--max-memory-mb", str(memory),
    ]
    if spin_penalty_shift > 0.0:
        command.extend(["--spin-penalty-shift", str(spin_penalty_shift)])
    print(f"RUN {r:.9f} A from R={read_r(restart):.9f} A -> {log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"R={r:.9f} A failed with code {result.returncode}; inspect {log}"
        )
    return point_h5


def run_side(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    rows = grid_rows(output)
    side_values = sorted(float(x["R_angstrom"]) for x in rows if x["side"] == args.side)
    local_values = sorted(float(x["R_angstrom"]) for x in rows if x["side"] == "local")
    if args.side == "short":
        side_values.reverse()
        restart_r = min(local_values)
    else:
        restart_r = max(local_values)
    restart = output / "points" / point_tag(restart_r) / "point.h5"
    if not restart.exists():
        raise FileNotFoundError(f"Missing local boundary seed: {restart}")
    script_dir = Path(__file__).resolve().parent
    for r in side_values:
        restart = run_one(script_dir, output, r, restart, args.max_memory_mb)
    print(f"PASS {args.side} side: {len(side_values)} new geometries", flush=True)


def status(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    rows = grid_rows(output)
    details = []
    counts = {"PASS": 0, "MISSING": 0, "REVIEW": 0, "CORRUPT": 0}
    for row in rows:
        r = float(row["R_angstrom"])
        point_dir = output / "points" / point_tag(r)
        state = "MISSING"
        detail = ""
        try:
            summary_path = point_dir / "summary.json"
            if summary_path.exists() and (point_dir / "point.h5").exists():
                state = json.loads(summary_path.read_text()).get("status", "REVIEW")
                if state not in counts:
                    state = "REVIEW"
        except Exception as exc:
            state, detail = "CORRUPT", f"{type(exc).__name__}: {exc}"
        counts[state] += 1
        details.append({**row, "status": state, "detail": detail})
    write_csv(output / "geometry_status.csv", details)
    result = {
        "point_count": len(rows), "counts": counts,
        "ready_for_analysis": counts["PASS"] == len(rows),
        "status_csv": str(output / "geometry_status.csv"),
    }
    print(json.dumps(result, indent=2))
    return result


def analyze(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    state = status(args)
    if not state["ready_for_analysis"]:
        raise SystemExit("Global analysis locked: not every geometry is PASS")
    protocol = json.loads((output / "protocol.json").read_text())
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u",
        str(Path(__file__).resolve().parent / "analyze_sa6_global_transport.py"),
        "--points", str(output / "points"), "--output", str(analysis_dir),
        "--anchor-R", str(protocol["anchor_R_angstrom"]),
        "--grid-csv", str(output / "geometry_grid.csv"),
        "--expected-R-min", str(protocol["R_window_angstrom"][0]),
        "--expected-R-max", str(protocol["R_window_angstrom"][1]),
    ]
    log = output / "logs" / "global_transport_analysis.stdout"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Global transport analysis failed; inspect {log}")
    summary = json.loads((analysis_dir / "transport_summary.json").read_text())
    production = {
        "status": summary["status"],
        "protocol": protocol,
        "transport_summary": summary,
    }
    (output / "production_summary.json").write_text(
        json.dumps(production, indent=2) + "\n"
    )
    print(json.dumps(production, indent=2))
    return summary


def refinable_edges(output: Path, minimum_step: float) -> list[tuple[float, float]]:
    with (output / "analysis" / "transport_edges.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    failed = []
    for row in rows:
        a, b = float(row["from_R_angstrom"]), float(row["to_R_angstrom"])
        bad = (
            float(row["active_min_sv"]) < 0.995
            or float(row["retained_min_sv"]) < 0.995
            or float(row["state_projector_min_sv"]) < 0.98
            or float(row["minimum_matched_abs_state_overlap"]) < 0.90
            or float(row["polar_transport_unitarity_error"]) > 1.0e-10
        )
        if bad and abs(b - a) / 2.0 >= minimum_step - 1.0e-12:
            failed.append((a, b))
    return failed


def insert_refinement_points(
    output: Path, edges: list[tuple[float, float]], memory_mb: int,
) -> list[float]:
    rows = grid_rows(output)
    local_values = sorted(float(row["R_angstrom"]) for row in rows if row["side"] == "local")
    local_min, local_max = min(local_values), max(local_values)
    existing = {round(float(row["R_angstrom"]), 10) for row in rows}
    new_values = []
    script_dir = Path(__file__).resolve().parent
    for a, b in edges:
        midpoint = round(0.5 * (a + b), 10)
        if midpoint in existing or local_min <= midpoint <= local_max:
            continue
        if midpoint < local_min:
            side, restart_r = "short", max(a, b)
        else:
            side, restart_r = "long", min(a, b)
        restart = output / "points" / point_tag(restart_r) / "point.h5"
        run_one(script_dir, output, midpoint, restart, memory_mb)
        rows.append({
            "R_angstrom": f"{midpoint:.10f}", "side": side,
            "grid_role": "adaptive_bridge", "plot_use": "0", "field_use": "0",
        })
        existing.add(midpoint)
        new_values.append(midpoint)
    rows.sort(key=lambda row: float(row["R_angstrom"]))
    write_csv(output / "geometry_grid.csv", rows)
    return new_values


def point_x_delta_gap(point_dir: Path) -> float:
    with (point_dir / "states.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    x_rows = [row for row in rows if row["local_assignment"].startswith("X ")]
    delta_rows = [row for row in rows if "Delta_g" in row["local_assignment"]]
    if len(x_rows) != 1 or len(delta_rows) != 2:
        raise RuntimeError(f"Cannot extract local X/Delta gap from {point_dir}")
    e_x = float(x_rows[0]["energy_Eh"])
    e_delta = sum(float(row["energy_Eh"]) for row in delta_rows) / 2.0
    return e_x - e_delta


def refine_crossing(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    production = json.loads((output / "production_summary.json").read_text())
    if production.get("status") != "PASS":
        raise SystemExit("Crossing refinement requires a PASS global transport")
    brackets = production["transport_summary"].get("crossing_brackets", [])
    if not 1 <= args.crossing_index <= len(brackets):
        raise SystemExit(
            f"crossing-index must lie in [1,{len(brackets)}]"
        )
    initial = dict(brackets[args.crossing_index - 1])
    low = float(initial["R_low_angstrom"])
    high = float(initial["R_high_angstrom"])
    gap_low = float(initial["gap_low_Eh"])
    gap_high = float(initial["gap_high_Eh"])
    if gap_low * gap_high > 0.0:
        raise RuntimeError("Selected interval does not bracket a sign change")

    rows = grid_rows(output)
    row_by_r = {round(float(row["R_angstrom"]), 10): row for row in rows}
    for edge in (low, high):
        row_by_r[round(edge, 10)]["field_use"] = "1"
        row_by_r[round(edge, 10)]["grid_role"] = "crossing_control"
    script_dir = Path(__file__).resolve().parent
    new_points = []
    iteration = 0
    while high - low > args.tolerance:
        iteration += 1
        if iteration > args.max_iterations:
            raise RuntimeError("Crossing refinement exceeded iteration limit")
        midpoint = round(0.5 * (low + high), 10)
        all_points = {
            read_r(path): path
            for path in (output / "points").glob("R_*/point.h5")
        }
        nearest_r = min(all_points, key=lambda value: abs(value - midpoint))
        point_h5 = run_one(
            script_dir, output, midpoint, all_points[nearest_r],
            args.max_memory_mb,
        )
        point_dir = point_h5.parent
        gap_mid = point_x_delta_gap(point_dir)
        key = round(midpoint, 10)
        if key not in row_by_r:
            row = {
                "R_angstrom": f"{midpoint:.10f}",
                "side": "long" if midpoint > ANCHOR_R else "short",
                "grid_role": "crossing_refinement",
                "plot_use": "0",
                "field_use": "1",
            }
            rows.append(row)
            row_by_r[key] = row
        else:
            row_by_r[key]["grid_role"] = "crossing_refinement"
            row_by_r[key]["field_use"] = "1"
        new_points.append({"R_angstrom": midpoint, "gap_Eh": gap_mid})
        if gap_low == 0.0 or gap_low * gap_mid <= 0.0:
            high, gap_high = midpoint, gap_mid
        else:
            low, gap_low = midpoint, gap_mid
        print(
            f"REFINE {iteration}: [{low:.10f},{high:.10f}] A; "
            f"width={high-low:.3e} A",
            flush=True,
        )

    rows.sort(key=lambda row: float(row["R_angstrom"]))
    write_csv(output / "geometry_grid.csv", rows)
    final_estimate = (
        low - gap_low * (high - low) / (gap_high - gap_low)
        if gap_high != gap_low else 0.5 * (low + high)
    )
    refinement = {
        "status": "POINTS_PASS_PENDING_GLOBAL_ANALYSIS",
        "crossing_index": args.crossing_index,
        "initial_bracket": initial,
        "new_points": new_points,
        "final_bracket": {
            "R_low_angstrom": low, "R_high_angstrom": high,
            "gap_low_Eh": gap_low, "gap_high_Eh": gap_high,
            "linear_crossing_estimate_angstrom": final_estimate,
            "width_angstrom": high - low,
        },
        "field_use_policy": (
            "original bracket endpoints plus every bisection point"
        ),
    }
    path = output / f"crossing_{args.crossing_index}_refinement.json"
    path.write_text(json.dumps(refinement, indent=2) + "\n")
    summary = analyze(args)
    if summary["status"] != "PASS":
        raise RuntimeError("Refined global transport failed continuity analysis")
    refinement["status"] = "PASS"
    refinement["global_transport_summary"] = summary
    path.write_text(json.dumps(refinement, indent=2) + "\n")
    print(json.dumps(refinement["final_bracket"], indent=2))


def run_both(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    processes = []
    for side in ("short", "long"):
        log = args.output.resolve() / "logs" / f"global_{side}_driver.stdout"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        command = [
            sys.executable, "-u", str(script), "run-side", "--side", side,
            "--output", str(args.output.resolve()),
            "--max-memory-mb", str(args.max_memory_mb),
        ]
        processes.append((side, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle, log))
    failures = []
    for side, process, handle, log in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append(f"{side} side code {code}; inspect {log}")
    if failures:
        raise RuntimeError("; ".join(failures))
    history = []
    history_path = args.output.resolve() / "adaptive_refinement_history.json"
    for iteration in range(args.max_edge_refinements + 1):
        summary = analyze(args)
        history.append({
            "iteration": iteration,
            "status": summary["status"],
            "number_of_points": summary["number_of_points"],
            "minimum_state_projector_sv": summary["minimum_state_projector_sv"],
            "minimum_matched_abs_state_overlap": summary["minimum_matched_abs_state_overlap"],
        })
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        if summary["status"] == "PASS":
            break
        edges = refinable_edges(args.output.resolve(), args.minimum_transport_step)
        if iteration == args.max_edge_refinements or not edges:
            raise RuntimeError(
                "Global transport remains REVIEW and has no further permitted "
                "edge bisection; inspect production_summary.json"
            )
        inserted = insert_refinement_points(
            args.output.resolve(), edges, args.max_memory_mb
        )
        history[-1]["inserted_midpoints_angstrom"] = inserted
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        if not inserted:
            raise RuntimeError("Failed edges could not be refined automatically")
    history_path.write_text(json.dumps(history, indent=2) + "\n")


def self_test(_: argparse.Namespace) -> None:
    local = decimal_grid(1.65, 1.75, 0.005) + [
        1.6525, 1.65265625, 1.652734375,
        1.6528125, 1.653125, 1.65375,
    ]
    rows = build_grid(
        local, DEFAULT_R_MIN, DEFAULT_R_MAX,
        DEFAULT_REPORT_STEP, DEFAULT_TRANSPORT_STEP,
    )
    assert len(rows) == 111
    assert sum(int(row["field_use"]) for row in rows) == 45
    assert sum(row["side"] == "short" for row in rows) == 26
    assert sum(row["side"] == "long" for row in rows) == 58
    assert max(
        b - a for a, b in zip(
            sorted(float(row["R_angstrom"]) for row in rows if row["side"] == "short"),
            sorted(float(row["R_angstrom"]) for row in rows if row["side"] == "short")[1:],
        )
    ) <= DEFAULT_TRANSPORT_STEP + 1.0e-12
    print("SELF-TEST PASS: 111 frames, 45 field geometries, 0.025 A maximum bridge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("self-test")
    p.set_defaults(func=self_test)
    p = sub.add_parser("prepare")
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--R-min", type=float, default=DEFAULT_R_MIN)
    p.add_argument("--R-max", type=float, default=DEFAULT_R_MAX)
    p.add_argument("--report-step", type=float, default=DEFAULT_REPORT_STEP)
    p.add_argument("--transport-step", type=float, default=DEFAULT_TRANSPORT_STEP)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run-side")
    p.add_argument("--side", choices=("short", "long"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-memory-mb", type=int, default=60000)
    p.set_defaults(func=run_side)
    p = sub.add_parser("run-both")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-memory-mb", type=int, default=60000)
    p.add_argument("--max-edge-refinements", type=int, default=3)
    p.add_argument("--minimum-transport-step", type=float, default=0.003125)
    p.set_defaults(func=run_both)
    p = sub.add_parser("status")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=status)
    p = sub.add_parser("analyze")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=analyze)
    p = sub.add_parser("refine-crossing")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crossing-index", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=1.0e-4)
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--max-memory-mb", type=int, default=60000)
    p.set_defaults(func=refine_crossing)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
