#!/usr/bin/env python3
"""Publication figure for the certified B2 Stage 2C cusp-response dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


EH_TO_MILLIHARTREE = 1000.0

COLORS = {
    "black": "#171717",
    "red": "#C74440",
    "blue": "#3976BA",
    "teal": "#2A9186",
    "amber": "#D58A17",
    "purple": "#7B61A8",
    "guide": "#8B8B8B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_xy(rows: list[dict[str, str]], ykey: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (as_float(row["R_Angstrom"]), as_float(row[ykey]))
        for row in rows
        if math.isfinite(as_float(row[ykey]))
    ]
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def split_plot(ax, x, y, rc, *, gap=2.5e-4, **kwargs):
    left = x < rc - gap
    right = x > rc + gap
    if np.any(left):
        ax.plot(x[left], y[left], **kwargs)
    if np.any(right):
        right_kwargs = dict(kwargs)
        right_kwargs.pop("label", None)
        ax.plot(x[right], y[right], **right_kwargs)


def style_axis(ax, panel: str, *, top_row: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.2,
        width=0.9,
        colors="black",
        labelcolor="black",
    )
    ax.minorticks_off()
    if top_row:
        ax.tick_params(labelbottom=False)
    ax.text(
        -0.105,
        1.015,
        f"({panel})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        clip_on=False,
        color="black",
    )


def add_crossing(ax, rc: float) -> None:
    ax.axvline(
        rc,
        color=COLORS["purple"],
        linestyle=(0, (3.0, 2.3)),
        linewidth=1.15,
        alpha=0.92,
        zorder=0,
    )


def validate_inputs(
    summary: dict,
    response_rows: list[dict[str, str]],
    axis_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> dict:
    failures: list[str] = []
    if summary.get("status") != "hybrid_response_pass":
        failures.append("Stage 2C summary is not hybrid_response_pass")
    if summary.get("warnings"):
        failures.append("Stage 2C summary contains warnings")
    expected = {
        "geometry_response_rows": 7,
        "axis_response_rows": 21,
        "paired_field_rows": 63,
    }
    actual = {
        "geometry_response_rows": len(response_rows),
        "axis_response_rows": len(axis_rows),
        "paired_field_rows": len(pair_rows),
    }
    for key, count in expected.items():
        if actual[key] != count or summary.get(key) != count:
            failures.append(f"{key}: expected {count}, got table={actual[key]}, summary={summary.get(key)}")
    for name, rows in {
        "geometry response": response_rows,
        "axis response": axis_rows,
        "pair audit": pair_rows,
    }.items():
        bad = [row for row in rows if row.get("status", "PASS") != "PASS"]
        if bad:
            failures.append(f"{name}: {len(bad)} non-PASS rows")
    cusp_rows = [r for r in response_rows if r["response_kind_x"] == "exact_crossing_abs_F_cusp"]
    if len(cusp_rows) != 1:
        failures.append(f"expected one exact crossing row, found {len(cusp_rows)}")
    max_xy = max(abs(as_float(row["xy_hybrid_relative_difference"])) for row in axis_rows)
    if max_xy > 3.0e-6:
        failures.append(f"x/y symmetry closure too large: {max_xy:.3e}")
    if failures:
        raise RuntimeError("Input certification failed:\n- " + "\n- ".join(failures))
    cusp = cusp_rows[0]
    return {
        "status": "PASS",
        "counts": actual,
        "maximum_xy_relative_difference": max_xy,
        "crossing_R_Angstrom": as_float(cusp["R_Angstrom"]),
        "mu_direct_perpendicular_au": as_float(cusp["mu_direct_perpendicular_au"]),
        "mu_relaxed_transport_perpendicular_au": as_float(
            cusp["mu_relaxed_transport_perpendicular_au"]
        ),
    }


def build_figure(
    stage2c_dir: Path,
    branch_map_path: Path,
    output_prefix: Path,
) -> dict:
    summary_path = stage2c_dir / "B2_stage2c_hybrid_summary.json"
    response_path = stage2c_dir / "B2_final_near_crossing_response.csv"
    axis_path = stage2c_dir / "hybrid_axis_response.csv"
    pair_path = stage2c_dir / "hybrid_pair_audit.csv"

    with summary_path.open() as handle:
        summary = json.load(handle)
    response_rows = read_csv(response_path)
    axis_rows = read_csv(axis_path)
    pair_rows = read_csv(pair_path)
    branch_rows = read_csv(branch_map_path)
    audit = validate_inputs(summary, response_rows, axis_rows, pair_rows)

    rc = audit["crossing_R_Angstrom"]
    cusp_row = next(r for r in response_rows if r["response_kind_x"] == "exact_crossing_abs_F_cusp")

    pes_rows = [
        row
        for row in branch_rows
        if 1.895 <= as_float(row["R_Angstrom"]) <= 2.005
        and row["X_status"] == "PASS"
        and row["A_B2u_status"] == "PASS"
        and row["A_B3u_status"] == "PASS"
    ]
    if len(pes_rows) < 10:
        raise RuntimeError("Insufficient certified PES rows in the crossing window")
    r_pes = np.asarray([as_float(row["R_Angstrom"]) for row in pes_rows])
    e_x = np.asarray([as_float(row["X_energy_Eh"]) for row in pes_rows])
    e_a = np.asarray([as_float(row["A_mean_energy_Eh"]) for row in pes_rows])
    e_c = 0.5 * (np.interp(rc, r_pes, e_x) + np.interp(rc, r_pes, e_a))

    r_resp = np.asarray([as_float(row["R_Angstrom"]) for row in response_rows])
    mu_direct = np.asarray([as_float(row["mu_direct_perpendicular_au"]) for row in response_rows])
    mu_relaxed = np.asarray(
        [as_float(row["mu_relaxed_transport_perpendicular_au"]) for row in response_rows]
    )
    ratio = np.asarray([as_float(row["orbital_response_renormalization_ratio"]) for row in response_rows])

    alpha_perp = np.asarray([as_float(row["alpha_x_final_au"]) for row in response_rows])
    alpha_parallel = np.asarray([as_float(row["alpha_z_final_au"]) for row in response_rows])
    alpha_iso = np.asarray([as_float(row["alpha_isotropic_final_au"]) for row in response_rows])
    metric_perp = np.asarray([as_float(row["metric_x_final_au"]) for row in response_rows])
    metric_parallel = np.asarray([as_float(row["metric_z_final_au"]) for row in response_rows])
    metric_mean = np.asarray([as_float(row["metric_cartesian_mean_final_au"]) for row in response_rows])

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.6,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 6.02))
    fig.subplots_adjust(left=0.105, right=0.895, bottom=0.105, top=0.945, wspace=0.43, hspace=0.15)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Keep the crossing guide vertically aligned in the left-hand column,
    # while retaining the global PES and local-response windows.
    response_xlim = (1.922, 1.978)
    rc_fraction = (rc - response_xlim[0]) / (response_xlim[1] - response_xlim[0])
    pes_left = 1.900
    pes_xlim = (pes_left, pes_left + (rc - pes_left) / rc_fraction)

    # (a) Certified symmetry-resolved PES crossing.
    ax_a.plot(
        r_pes,
        (e_x - e_c) * EH_TO_MILLIHARTREE,
        color=COLORS["black"],
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$X\,^3\Sigma_g^-$",
    )
    ax_a.plot(
        r_pes,
        (e_a - e_c) * EH_TO_MILLIHARTREE,
        color=COLORS["red"],
        marker="s",
        markersize=3.35,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$A\,^3\Pi_u$",
    )
    add_crossing(ax_a, rc)
    ax_a.text(
        rc + 0.0030,
        0.075,
        rf"$R_c={rc:.3g}\ \mathrm{{\AA}}$",
        transform=ax_a.get_xaxis_transform(),
        rotation=0,
        ha="left",
        va="bottom",
        color="black",
        fontsize=7.8,
    )
    ax_a.set_xlim(*pes_xlim)
    ax_a.set_ylabel(r"Energy, $E-E_c$ (m$E_h$)")
    ax_a.legend(loc="upper left", frameon=False, handlelength=2.1, borderaxespad=0.35)
    style_axis(ax_a, "a", top_row=True)

    # (b) Direct transition moment and relaxed transported diagnostic.
    ax_b.plot(
        r_resp,
        mu_direct,
        color=COLORS["black"],
        marker="o",
        markersize=3.7,
        label=r"direct $\mu_\perp$",
    )
    ax_b.plot(
        r_resp,
        mu_relaxed,
        color=COLORS["teal"],
        marker="s",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.95,
        label=r"relaxed $\widetilde{\mu}_\perp$",
    )
    add_crossing(ax_b, rc)
    ax_b.set_xlim(*response_xlim)
    ax_b.set_ylim(0.185, 0.266)
    ax_b.set_ylabel("Transition moment (a.u.)")
    ax_b.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.35)
    ax_br = ax_b.twinx()
    ax_br.plot(
        r_resp,
        ratio,
        color=COLORS["purple"],
        linestyle=(0, (2.0, 1.8)),
        marker="D",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.85,
        label="relaxation ratio",
    )
    ax_br.set_ylim(1.258, 1.291)
    ax_br.set_ylabel(r"$\widetilde{\mu}_\perp/\mu_\perp$", color="black")
    ax_br.tick_params(axis="y", direction="in", right=True, left=False, length=4.2, width=0.9, colors="black")
    ax_br.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False)
    ax_br.minorticks_off()
    ax_br.spines["right"].set_linewidth(0.9)
    ax_br.spines["right"].set_color("black")
    ax_br.spines["left"].set_visible(False)
    ax_br.spines["top"].set_visible(False)
    ax_br.spines["bottom"].set_visible(False)
    ax_br.legend(
        loc="center right",
        bbox_to_anchor=(0.985, 0.43),
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.0,
    )
    style_axis(ax_b, "b", top_row=True)
    # The twin axis exclusively owns the right-hand ticks and spine.
    ax_b.tick_params(axis="y", right=False)
    ax_b.spines["right"].set_visible(False)

    # (c) Hybrid polarizability; the exact crossing is a |F| cusp.
    split_plot(
        ax_c,
        r_resp,
        alpha_perp,
        rc,
        color=COLORS["blue"],
        marker="o",
        markersize=3.8,
        label=r"$\alpha_\perp$",
    )
    split_plot(
        ax_c,
        r_resp,
        alpha_parallel,
        rc,
        color=COLORS["amber"],
        linestyle="--",
        marker="s",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$\alpha_\parallel$",
    )
    split_plot(
        ax_c,
        r_resp,
        alpha_iso,
        rc,
        color=COLORS["teal"],
        linestyle=":",
        marker="D",
        markersize=3.3,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$\bar{\alpha}$",
    )
    add_crossing(ax_c, rc)
    ax_c.set_xlim(*response_xlim)
    ax_c.set_ylim(0, 1740)
    ax_c.set_xlabel(r"B--B distance, $R$ ($\mathrm{\AA}$)")
    ax_c.set_ylabel("Polarizability (a.u.)")
    ax_c.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.35)
    style_axis(ax_c, "c")

    # (d) State-projector quantum metric.
    split_plot(
        ax_d,
        r_resp,
        metric_perp,
        rc,
        color=COLORS["blue"],
        marker="o",
        markersize=3.8,
        label=r"$g_\perp^{(P)}$",
    )
    split_plot(
        ax_d,
        r_resp,
        metric_parallel,
        rc,
        color=COLORS["amber"],
        linestyle="--",
        marker="s",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$g_\parallel^{(P)}$",
    )
    split_plot(
        ax_d,
        r_resp,
        metric_mean,
        rc,
        color=COLORS["teal"],
        linestyle=":",
        marker="D",
        markersize=3.3,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"$\bar{g}^{(P)}$",
    )
    add_crossing(ax_d, rc)
    ax_d.set_xlim(*response_xlim)
    ax_d.set_yscale("log")
    ax_d.set_ylim(8.0e1, 4.5e7)
    ax_d.set_xlabel(r"B--B distance, $R$ ($\mathrm{\AA}$)")
    ax_d.set_ylabel(r"Quantum metric, $g^{(P)}$ (a.u.)")
    ax_d.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.35)
    style_axis(ax_d, "d")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    finite_alpha_parallel = alpha_parallel[np.isfinite(alpha_parallel)]
    finite_metric_parallel = metric_parallel[np.isfinite(metric_parallel)]
    result = {
        "status": "PASS",
        "figure": {
            "pdf": str(output_prefix.with_suffix(".pdf").resolve()),
            "png": str(output_prefix.with_suffix(".png").resolve()),
            "svg": str(output_prefix.with_suffix(".svg").resolve()),
        },
        "source_files": {
            "stage2c_summary": str(summary_path.resolve()),
            "geometry_response": str(response_path.resolve()),
            "axis_response": str(axis_path.resolve()),
            "pair_audit": str(pair_path.resolve()),
            "stage1b_branch_map": str(branch_map_path.resolve()),
        },
        "certification": audit,
        "reported_quantities": {
            "crossing_R_Angstrom": rc,
            "mu_direct_perpendicular_au": as_float(cusp_row["mu_direct_perpendicular_au"]),
            "mu_relaxed_transport_perpendicular_au": as_float(
                cusp_row["mu_relaxed_transport_perpendicular_au"]
            ),
            "cusp_slope_direct_perpendicular_au": as_float(
                cusp_row["cusp_slope_direct_perpendicular_au"]
            ),
            "relaxation_ratio_range": [float(np.nanmin(ratio)), float(np.nanmax(ratio))],
            "alpha_parallel_range_au": [
                float(np.min(finite_alpha_parallel)),
                float(np.max(finite_alpha_parallel)),
            ],
            "metric_parallel_range_au": [
                float(np.min(finite_metric_parallel)),
                float(np.max(finite_metric_parallel)),
            ],
        },
        "plotting_policy": {
            "exact_crossing_response": "left as a cusp; no finite alpha or metric value plotted",
            "direct_vs_relaxed": "shown separately; relaxed/direct ratio is diagnostic only",
            "line_bridging_across_crossing": False,
            "minor_ticks": False,
            "left_column_crossing_axis_fraction": rc_fraction,
        },
    }
    provenance = output_prefix.with_name(output_prefix.name + "_provenance.json")
    provenance.write_text(json.dumps(result, indent=2) + "\n")

    audit_csv = output_prefix.with_name(output_prefix.name + "_audit.csv")
    with audit_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "value", "status"])
        writer.writerow(["Stage 2C summary", summary["status"], "PASS"])
        writer.writerow(["paired signed-field rows", len(pair_rows), "PASS"])
        writer.writerow(["axis-response rows", len(axis_rows), "PASS"])
        writer.writerow(["near-crossing geometries", len(response_rows), "PASS"])
        writer.writerow(["maximum x/y relative difference", f"{audit['maximum_xy_relative_difference']:.9e}", "PASS"])
        writer.writerow(["crossing R (Angstrom)", f"{rc:.13f}", "PASS"])
        writer.writerow(["left-column crossing alignment fraction", f"{rc_fraction:.12f}", "PASS"])
        writer.writerow(["exact cusp left unconnected", "true", "PASS"])
        writer.writerow(["direct and relaxed moments separated", "true", "PASS"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2c-dir", type=Path, required=True)
    parser.add_argument("--branch-map", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = build_figure(args.stage2c_dir, args.branch_map, args.output_prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
