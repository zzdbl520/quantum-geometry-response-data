#!/usr/bin/env python3
"""Plot SS-vs-SA2 response and SS/SA4 continuity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ho2_core import read_csv
from protocol_core import load_metadata, validation_root


def contiguous_groups(rows):
    groups = []
    current = []
    for row in sorted(rows, key=lambda item: int(item["source_index"])):
        index = int(row["source_index"])
        if current and index != int(current[-1]["source_index"]) + 1:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    metadata = load_metadata(workdir)
    analysis = validation_root(workdir) / "analysis"
    summary = read_csv(analysis / "ss_qgeom_summary.csv")
    audit = read_csv(analysis / "zero_protocol_audit.csv")
    gate = json.loads((analysis / "protocol_validation_gate.json").read_text())
    if not summary:
        raise SystemExit("Empty SS summary; run analyze_protocol_validation.py")
    pilot_path = Path(metadata["pilot_workdir"]) / "analysis" / "qgeom_summary.csv"
    pilot = {
        int(row["source_index"]): row for row in read_csv(pilot_path)
    } if pilot_path.exists() else {}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), constrained_layout=True)
    colors = {"alpha": "#0068a8", "metric": "#d45500", "delta": "#009e73"}

    for group_number, group in enumerate(contiguous_groups(summary)):
        s = values(group, "s_cart_A")
        alpha = values(group, "alpha_parallel_au")
        error = values(group, "alpha_fit_uncertainty_au")
        metric = values(group, "g_FF_au")
        delta = values(group, "Delta_geom_Eh")
        label = "SS-CASSCF" if group_number == 0 else None
        axes[0, 0].errorbar(
            s, alpha, yerr=error, color=colors["alpha"], marker="o",
            lw=1.5, ms=4, capsize=2, label=label,
        )
        axes[0, 1].plot(
            s, metric, color=colors["metric"], marker="o", lw=1.5,
            ms=4, label=label,
        )
        axes[1, 0].plot(
            s, delta, color=colors["delta"], marker="o", lw=1.5,
            ms=4, label=label,
        )
        if pilot:
            pilot_group = [pilot[int(row["source_index"])] for row in group]
            axes[0, 0].plot(
                s, values(pilot_group, "alpha_parallel_au"), "--",
                color="#777777", lw=1.2,
                label="SA2-CASSCF" if group_number == 0 else None,
            )
            axes[0, 1].plot(
                s, values(pilot_group, "g_FF_au"), "--",
                color="#777777", lw=1.2,
            )
            axes[1, 0].plot(
                s, values(pilot_group, "Delta_geom_Eh"), "--",
                color="#777777", lw=1.2,
            )

    status_by_index = gate.get("target_peak_status", {})
    summary_map = {int(row["source_index"]): row for row in summary}
    for index, status in status_by_index.items():
        idx = int(index)
        if idx not in summary_map:
            continue
        row = summary_map[idx]
        marker = "o" if status == "ROBUST_MAXIMUM" else "o"
        face = colors["alpha"] if status == "ROBUST_MAXIMUM" else "white"
        axes[0, 0].plot(
            float(row["s_cart_A"]), float(row["alpha_parallel_au"]),
            marker=marker, ms=7, markerfacecolor=face,
            markeredgecolor=colors["alpha"], zorder=5,
        )

    axes[0, 0].set_ylabel(r"$\alpha_{\parallel}$ (a.u.)")
    axes[0, 1].set_ylabel(r"$g_{FF}$ (a.u.)")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel(r"$\Delta_{\mathrm{geom}}$ ($E_{\mathrm{h}}$)")
    axes[1, 0].set_xlabel(r"IRC arc coordinate, $s$ ($\mathrm{\AA}$)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for protocol, color, marker in (("ss", "#6a3d9a", "o"), ("sa4", "#cc79a7", "s")):
        subset = [
            row for row in audit
            if row["protocol"] == protocol and row["status"] != "SEED"
        ]
        x = np.asarray([int(row["source_index"]) for row in subset])
        key = (
            "parent_ground_best_overlap"
            if protocol == "ss"
            else "parent_subspace_sv_min"
        )
        y = values(subset, key)
        ax.plot(x, y, color=color, marker=marker, ms=3.5, lw=1.2, label=protocol.upper())
    ax.axhline(0.75, color="#555555", ls="--", lw=1.0)
    ax.set_xlabel("Source index")
    ax.set_ylabel("Adjacent-state continuity")
    ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False, fontsize=8)

    axes[0, 1].set_xlabel(r"IRC arc coordinate, $s$ ($\mathrm{\AA}$)")
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=10)
        ax.tick_params(which="major", length=5, width=1.0, colors="black")
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_color("black")

    output_pdf = analysis / "HO2_protocol_validation.pdf"
    output_png = analysis / "HO2_protocol_validation.png"
    fig.savefig(output_pdf, dpi=300)
    fig.savefig(output_png, dpi=300)
    print(f"PDF: {output_pdf}", flush=True)
    print(f"PNG: {output_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
