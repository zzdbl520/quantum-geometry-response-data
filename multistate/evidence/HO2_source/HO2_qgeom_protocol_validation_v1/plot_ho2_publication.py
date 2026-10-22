#!/usr/bin/env python3
"""Compact main-text and SI HO2 response figures from the v1.2 summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ho2_core import read_csv
from protocol_core import validation_root


PURPLE = "#6A3D9A"
BLUE = "#1F78B4"
ORANGE = "#D95F02"
GREEN = "#1B9E77"
BLACK = "#1A1A1A"
GRAY = "#6B6B6B"


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def select(rows, first, last):
    return [
        row for row in rows
        if first <= int(row["source_index"]) <= last
    ]


def normalized_irc_coordinate(rows, source_count):
    """Map the original ordered IRC frames once onto the common [0, 1] axis."""
    if source_count < 2:
        raise ValueError("The IRC dataset must contain at least two geometries")
    return np.asarray(
        [int(row["source_index"]) / (source_count - 1) for row in rows],
        dtype=float,
    )


def robust_maximum_positions(rows):
    lower = values(rows, "plateau_reconstructed_ground_alpha_lower_au")
    upper = values(rows, "plateau_reconstructed_ground_alpha_upper_au")
    answer = []
    for i in range(1, len(rows) - 1):
        if lower[i] > max(upper[i - 1], upper[i + 1]):
            answer.append(i)
    return answer


def finite_contiguous_segments(rows, y):
    """Split finite values into source-index-contiguous plotting segments."""
    indices = np.asarray([int(row["source_index"]) for row in rows], dtype=int)
    finite_positions = np.flatnonzero(np.isfinite(y))
    if finite_positions.size == 0:
        return []
    breaks = np.flatnonzero(
        (np.diff(finite_positions) != 1)
        | (np.diff(indices[finite_positions]) != 1)
    ) + 1
    return [segment for segment in np.split(finite_positions, breaks) if segment.size]


def plot_ss_segments(ax, rows, x, y):
    """Do not imply interpolation across uncomputed SS geometries."""
    for number, segment in enumerate(finite_contiguous_segments(rows, y)):
        ax.plot(
            x[segment], y[segment], "s-", color=GREEN,
            markerfacecolor="white", lw=1.25, ms=4.2,
            label="SS root" if number == 0 else "_nolegend_",
        )


def style_axis(ax):
    ax.tick_params(
        which="major", direction="in", top=True, right=True,
        colors=BLACK, width=1.0, length=4.5,
    )
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(1.05)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)


def add_peak_guides(ax, rows, x):
    y = values(rows, "plateau_reconstructed_ground_alpha_median_au")
    top = ax.get_ylim()[1]
    peak_labels = ("Peak I", "Peak II")
    for number, position in enumerate(robust_maximum_positions(rows)):
        ax.vlines(
            x[position], y[position], top,
            color=BLACK, lw=0.9, linestyles=(0, (3, 2)), zorder=0,
        )
        ax.scatter(
            [x[position]], [y[position]], marker="v", s=47,
            facecolor="white", edgecolor=BLACK, linewidth=1.0, zorder=6,
        )
        ax.annotate(
            peak_labels[number] if number < len(peak_labels) else f"Peak {number + 1}",
            xy=(x[position], y[position]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom", color=BLACK, fontsize=8.5,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--output-prefix", default="HO2_response_mechanisms")
    parser.add_argument("--si-output-prefix", default="HO2_SI_late_path_control")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    analysis = validation_root(workdir) / "analysis" / "manifold"
    rows = sorted(
        read_csv(analysis / "manifold_summary.csv"),
        key=lambda row: int(row["grid_position"]),
    )
    if not rows:
        raise SystemExit("Empty manifold summary; run v1.2 analysis first")
    required = {
        "plateau_reconstructed_ground_alpha_median_au",
        "plateau_external_ground_alpha_median_au",
        "plateau_internal_manifold_ground_alpha_median_au",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit("Summary predates v1.2; missing: " + ", ".join(missing))

    early = select(rows, 2, 11)
    artifact = select(rows, 15, 18)
    if len(early) != 10 or len(artifact) != 4:
        raise SystemExit(
            "Expected complete source windows 2--11 and 15--18; "
            f"found {len(early)} and {len(artifact)} rows"
        )
    source_data = np.load(Path(__file__).resolve().parent / "data" / "01_irc.npz")
    source_count = len(source_data["R"])
    source_data.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.linewidth": 1.05,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "text.color": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "savefig.transparent": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.15))

    # (a) Root-resolved versus continuous-manifold response.
    ax = axes[0]
    x = normalized_irc_coordinate(early, source_count)
    total = values(early, "plateau_reconstructed_ground_alpha_median_au")
    lower = values(early, "plateau_reconstructed_ground_alpha_lower_au")
    upper = values(early, "plateau_reconstructed_ground_alpha_upper_au")
    sa2 = values(early, "sa2_ground_alpha_au")
    ss = values(early, "ss_ground_alpha_au")
    ax.fill_between(x, lower, upper, color=PURPLE, alpha=0.15, linewidth=0)
    ax.plot(
        x, total, "o-", color=PURPLE, markerfacecolor="white",
        markeredgewidth=1.15, lw=1.8, ms=4.7, label="rank-2 manifold",
    )
    ax.plot(x, sa2, "--", color=GRAY, lw=1.35, label="SA2 root")
    plot_ss_segments(ax, early, x, ss)
    add_peak_guides(ax, early, x)
    ax.set_xlabel(r"Normalized IRC coordinate, $\xi_{\mathrm{IRC}}$")
    ax.set_ylabel(r"Polarizability (a.u.)")
    ax.legend(frameon=False, loc="best", handlelength=2.2)
    style_axis(ax)

    # (b) Additive response decomposition that creates the two maxima.
    ax = axes[1]
    external = values(early, "plateau_external_ground_alpha_median_au")
    internal = values(early, "plateau_internal_manifold_ground_alpha_median_au")
    ax.plot(
        x, external, "s-", color=BLUE, markerfacecolor="white",
        markeredgewidth=1.1, lw=1.7, ms=4.5, label=r"external $A_{00}$",
    )
    ax.set_xlabel(r"Normalized IRC coordinate, $\xi_{\mathrm{IRC}}$")
    ax.set_ylabel(r"External response (a.u.)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    twin = ax.twinx()
    twin.plot(
        x, internal, "^-", color=ORANGE, markerfacecolor="white",
        markeredgewidth=1.1, lw=1.7, ms=4.7,
        label=r"internal $2|M_{01}|^2/\Delta_{10}$",
    )
    twin.set_ylabel(r"Internal response (a.u.)", color=ORANGE)
    twin.tick_params(
        axis="y", which="major", direction="in", right=True,
        colors=ORANGE, width=1.0, length=4.5,
    )
    twin.minorticks_off()
    handles = ax.get_lines() + twin.get_lines()
    ax.legend(
        handles, [line.get_label() for line in handles],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        ncol=2, borderaxespad=0.0, handlelength=1.7,
        handletextpad=0.45, columnspacing=0.85, fontsize=7.7,
    )
    for position in robust_maximum_positions(early):
        ax.axvline(x[position], color=BLACK, lw=0.9, ls=(0, (3, 2)), zorder=0)
    style_axis(ax)
    twin.spines["right"].set_color(BLACK)
    twin.spines["right"].set_linewidth(1.05)
    twin.spines["top"].set_color(BLACK)
    twin.spines["top"].set_linewidth(1.05)

    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(
            -0.18, 1.04, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, fontweight="bold",
            color=BLACK,
        )

    fig.subplots_adjust(
        left=0.095, right=0.905, bottom=0.19, top=0.94,
        wspace=0.36,
    )
    output_pdf = analysis / f"{args.output_prefix}.pdf"
    output_png = analysis / f"{args.output_prefix}.png"
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight")
    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    # Standalone SI control: removal of the apparent late-path maximum.
    fig_si, ax = plt.subplots(figsize=(3.65, 3.05))
    x3 = normalized_irc_coordinate(artifact, source_count)
    total3 = values(artifact, "plateau_reconstructed_ground_alpha_median_au")
    lower3 = values(artifact, "plateau_reconstructed_ground_alpha_lower_au")
    upper3 = values(artifact, "plateau_reconstructed_ground_alpha_upper_au")
    sa23 = values(artifact, "sa2_ground_alpha_au")
    ss3 = values(artifact, "ss_ground_alpha_au")
    ax.fill_between(x3, lower3, upper3, color=PURPLE, alpha=0.15, linewidth=0)
    ax.plot(
        x3, total3, "o-", color=PURPLE, markerfacecolor="white",
        markeredgewidth=1.15, lw=1.8, ms=4.7, label="rank-2 manifold",
    )
    ax.plot(
        x3, sa23, "x--", color=GRAY, lw=1.35, ms=5.0,
        label="SA2 root",
    )
    plot_ss_segments(ax, artifact, x3, ss3)
    collapse = int(np.nanargmin(sa23))
    ax.annotate(
        "SA2 solution\nchange",
        xy=(x3[collapse], sa23[collapse]),
        xytext=(-14, 18), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.3, color=BLACK,
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": BLACK},
    )
    ax.set_xlabel(r"Normalized IRC coordinate, $\xi_{\mathrm{IRC}}$")
    ax.set_ylabel(r"Polarizability (a.u.)")
    ax.legend(frameon=False, loc="best", handlelength=2.2)
    style_axis(ax)
    fig_si.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.96)
    si_output_pdf = analysis / f"{args.si_output_prefix}.pdf"
    si_output_png = analysis / f"{args.si_output_prefix}.png"
    fig_si.savefig(si_output_pdf, dpi=600, bbox_inches="tight")
    fig_si.savefig(si_output_png, dpi=400, bbox_inches="tight")
    plt.close(fig_si)
    print(output_pdf)
    print(output_png)
    print(si_output_pdf)
    print(si_output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
