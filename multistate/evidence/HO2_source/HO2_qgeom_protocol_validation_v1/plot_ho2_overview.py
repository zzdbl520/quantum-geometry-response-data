#!/usr/bin/env python3
"""Create the unified HO2 Figure 1 and a compact TOC graphic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ho2_core import read_csv
from protocol_core import validation_root


PURPLE = "#6A3D9A"
BLUE = "#1F78B4"
ORANGE = "#D95F02"
RED = "#D64B3C"
BLACK = "#1A1A1A"
LIGHT_PURPLE = "#D8C9E8"


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def normalized_coordinate(rows, source_count):
    return np.asarray(
        [int(row["source_index"]) / (source_count - 1) for row in rows],
        dtype=float,
    )


def robust_maxima(rows):
    lower = numeric(rows, "plateau_reconstructed_ground_alpha_lower_au")
    upper = numeric(rows, "plateau_reconstructed_ground_alpha_upper_au")
    return [
        index for index in range(1, len(rows) - 1)
        if lower[index] > max(upper[index - 1], upper[index + 1])
    ]


def style_axis(ax):
    ax.tick_params(
        which="major", direction="in", top=True, right=True,
        colors=BLACK, width=1.0, length=4.5,
    )
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(1.05)


def panel_label(ax, label):
    ax.text(
        -0.12, 1.04, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=11, fontweight="bold", color=BLACK,
    )


def draw_atom(ax, x, y, element, radius, transform=None, zorder=5):
    transform = transform or ax.transAxes
    face = RED if element == "O" else "white"
    diameter_points = 2.0 * radius * ax.figure.get_figwidth() * 72.0
    ax.scatter(
        [x], [y], transform=transform, s=diameter_points**2,
        marker="o", facecolor=face, edgecolor=BLACK, linewidth=1.0,
        clip_on=False, zorder=zorder,
    )
    ax.text(
        x, y, element, transform=transform, ha="center", va="center",
        color="white" if element == "O" else BLACK,
        fontsize=7.0 if element == "O" else 6.3,
        fontweight="bold", clip_on=False, zorder=zorder + 1,
    )


def draw_species(ax, center_x, center_y, kind, scale=1.0):
    """Draw collinear O...O-H or O-O...H in axes coordinates."""
    o_radius = 0.028 * scale
    h_radius = 0.021 * scale
    if kind == "oh_plus_o":
        positions = [center_x - 0.12 * scale, center_x + 0.01 * scale,
                     center_x + 0.09 * scale]
        bonds = [(1, 2, "solid"), (0, 1, "dotted")]
    elif kind == "o2_plus_h":
        positions = [center_x - 0.09 * scale, center_x - 0.01 * scale,
                     center_x + 0.12 * scale]
        bonds = [(0, 1, "double"), (1, 2, "dotted")]
    else:
        raise ValueError(kind)
    for left, right, bond_kind in bonds:
        x0, x1 = positions[left], positions[right]
        if bond_kind == "solid":
            ax.plot([x0, x1], [center_y, center_y], transform=ax.transAxes,
                    color=BLACK, lw=1.8, clip_on=False, zorder=3)
        elif bond_kind == "double":
            offset = 0.008 * scale
            for dy in (-offset, offset):
                ax.plot([x0, x1], [center_y + dy, center_y + dy],
                        transform=ax.transAxes, color=BLACK, lw=1.35,
                        clip_on=False, zorder=3)
        else:
            ax.plot([x0, x1], [center_y, center_y], transform=ax.transAxes,
                    color="#777777", lw=1.1, ls=(0, (1.5, 2.0)),
                    clip_on=False, zorder=2)
    draw_atom(ax, positions[0], center_y, "O", o_radius)
    draw_atom(ax, positions[1], center_y, "O", o_radius)
    draw_atom(ax, positions[2], center_y, "H", h_radius)


def create_figure1(rows, source_count, output_dir):
    import matplotlib.pyplot as plt

    early = [row for row in rows if 2 <= int(row["source_index"]) <= 11]
    by_source = {int(row["source_index"]): row for row in rows}
    x = normalized_coordinate(early, source_count)
    total = numeric(early, "plateau_reconstructed_ground_alpha_median_au")
    lower = numeric(early, "plateau_reconstructed_ground_alpha_lower_au")
    upper = numeric(early, "plateau_reconstructed_ground_alpha_upper_au")
    maxima = robust_maxima(early)

    fig = plt.figure(figsize=(8.35, 5.85))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    top = fig.add_subplot(grid[0, :])
    left = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[1, 1])

    top.fill_between(x, lower, upper, color=PURPLE, alpha=0.16, linewidth=0)
    top.plot(
        x, total, "o-", color=PURPLE, lw=2.0, ms=5.0,
        markerfacecolor="white", markeredgewidth=1.2,
    )
    peak_labels = ("M1", "M2")
    for label, position in zip(peak_labels, maxima):
        top.scatter(
            [x[position]], [total[position]], marker="v", s=62,
            facecolor="white", edgecolor=BLACK, linewidth=1.0, zorder=6,
        )
        top.annotate(
            label, xy=(x[position], total[position]),
            xytext=(0, 11), textcoords="offset points",
            ha="center", va="bottom", fontsize=9.3, color=BLACK,
        )
        top.vlines(
            x[position], total[position], top.get_ylim()[1],
            color=BLACK, lw=0.9, linestyles=(0, (3, 2)), zorder=0,
        )
    top.text(
        0.02, 0.93,
        r"$\mathrm{OH+O}\;\rightleftharpoons\;\mathrm{H+O_2}$",
        transform=top.transAxes, ha="left", va="top", fontsize=11.0,
    )
    top.set_xlabel(r"Normalized IRC coordinate, $\xi_{\mathrm{IRC}}$")
    top.set_ylabel(r"Ground-state polarizability (a.u.)")
    style_axis(top)
    panel_label(top, "(a)")

    # M1: the internal retained-manifold term is a large fraction of alpha_0.
    m1 = by_source[5]
    external = float(m1["plateau_external_ground_alpha_median_au"])
    internal = float(m1["plateau_internal_manifold_ground_alpha_median_au"])
    gap = float(m1["internal_gap_zero_Eh"])
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.axis("off")
    left.text(
        0.0, 1.02, "(b)  M1: low-gap amplification",
        transform=left.transAxes, ha="left", va="bottom",
        fontsize=10.2, fontweight="bold", color=BLACK,
    )
    bar_x, bar_w = 0.10, 0.22
    scale = 0.72 / (external + internal)
    left.bar(
        [bar_x], [external * scale], width=bar_w, bottom=0.12,
        color=BLUE, edgecolor="white", linewidth=0.8,
    )
    left.bar(
        [bar_x], [internal * scale], width=bar_w,
        bottom=0.12 + external * scale,
        color=ORANGE, edgecolor="white", linewidth=0.8,
    )
    left.plot(
        [bar_x - bar_w / 2, bar_x + bar_w / 2,
         bar_x + bar_w / 2, bar_x - bar_w / 2, bar_x - bar_w / 2],
        [0.12, 0.12, 0.84, 0.84, 0.12], color=PURPLE, lw=1.5,
    )
    left.text(bar_x, 0.12 + external * scale / 2, r"$A_{00}$",
              ha="center", va="center", color="white", fontsize=9.3)
    left.text(
        bar_x, 0.12 + external * scale + internal * scale / 2,
        r"$2|M_{01}|^2/\Delta_{10}$", ha="center", va="center",
        color="white", fontsize=7.8,
    )
    left.text(
        bar_x, 0.07, rf"{internal / (external + internal):.0%} internal",
        ha="center", va="top", fontsize=8.2, color=BLACK,
    )
    level_x0, level_x1 = 0.48, 0.78
    y0, y1 = 0.36, 0.53
    left.plot([level_x0, level_x1], [y0, y0], color=BLACK, lw=2.0)
    left.plot([level_x0, level_x1], [y1, y1], color=BLACK, lw=2.0)
    left.text(level_x1 + 0.025, y0, r"$E_0$", va="center", fontsize=9.0)
    left.text(level_x1 + 0.025, y1, r"$E_1$", va="center", fontsize=9.0)
    left.annotate(
        "", xy=(0.62, y1 - 0.01), xytext=(0.62, y0 + 0.01),
        arrowprops={"arrowstyle": "<->", "lw": 1.3, "color": ORANGE},
    )
    left.text(
        0.65, (y0 + y1) / 2,
        rf"$\Delta_{{10}}={gap:.4f}\ E_h$",
        ha="left", va="center", fontsize=8.3, color=ORANGE,
    )
    left.text(
        0.48, 0.74, r"$\alpha_0=A_{00}+2|M_{01}|^2/\Delta_{10}$",
        ha="left", va="center", fontsize=10.0, color=BLACK,
    )

    # M2: plot actual component changes over sources 8--10.
    local = [by_source[index] for index in (8, 9, 10)]
    x2 = normalized_coordinate(local, source_count)
    ext = numeric(local, "plateau_external_ground_alpha_median_au")
    inte = numeric(local, "plateau_internal_manifold_ground_alpha_median_au")
    tot = numeric(local, "plateau_reconstructed_ground_alpha_median_au")
    ext -= ext[0]
    inte -= inte[0]
    tot -= tot[0]
    right.axhline(0.0, color="#AAAAAA", lw=0.8, zorder=0)
    right.plot(x2, ext, "s-", color=BLUE, markerfacecolor="white",
               lw=1.7, ms=4.8, label=r"$\Delta A_{00}$")
    right.plot(x2, inte, "^-", color=ORANGE, markerfacecolor="white",
               lw=1.7, ms=5.0, label=r"$\Delta\alpha_{\mathrm{int}}$")
    right.plot(x2, tot, "o-", color=PURPLE, markerfacecolor="white",
               lw=2.0, ms=5.0, label=r"$\Delta\alpha_0$")
    right.axvline(x2[1], color=BLACK, lw=0.9, ls=(0, (3, 2)))
    right.annotate(
        r"$\partial_{\xi}\alpha_0=0$",
        xy=(x2[1], tot[1]), xytext=(0, 13), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.7, color=BLACK,
    )
    right.set_xlabel(r"Normalized IRC coordinate, $\xi_{\mathrm{IRC}}$")
    right.set_ylabel(r"Change from source 8 (a.u.)")
    label_x = x2[-1] + 0.0015
    right.text(label_x, ext[-1], r"$\Delta A_{00}$", color=BLUE,
               ha="left", va="center", fontsize=8.2)
    right.text(label_x, inte[-1], r"$\Delta\alpha_{\mathrm{int}}$", color=ORANGE,
               ha="left", va="center", fontsize=8.2)
    right.text(label_x, tot[-1], r"$\Delta\alpha_0$", color=PURPLE,
               ha="left", va="center", fontsize=8.2)
    right.set_xlim(x2[0] - 0.002, x2[-1] + 0.010)
    right.set_title(
        "(c)  M2: external–internal compensation",
        loc="left", fontsize=10.2, fontweight="bold", pad=9,
    )
    style_axis(right)

    fig.subplots_adjust(
        left=0.09, right=0.975, bottom=0.10, top=0.96,
        hspace=0.48, wspace=0.34,
    )
    pdf = output_dir / "HO2_Figure1_unified.pdf"
    png = output_dir / "HO2_Figure1_unified.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def create_toc(rows, source_count, output_dir):
    import matplotlib.pyplot as plt

    early = [row for row in rows if 2 <= int(row["source_index"]) <= 11]
    x = normalized_coordinate(early, source_count)
    total = numeric(early, "plateau_reconstructed_ground_alpha_median_au")
    maxima = robust_maxima(early)
    x_scaled = 0.31 + 0.44 * (x - x.min()) / (x.max() - x.min())
    y_scaled = 0.35 + 0.30 * (total - total.min()) / (total.max() - total.min())

    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_species(ax, 0.15, 0.83, "oh_plus_o", scale=0.80)
    draw_species(ax, 0.85, 0.83, "o2_plus_h", scale=0.80)
    ax.annotate(
        "", xy=(0.69, 0.83), xytext=(0.31, 0.83),
        arrowprops={"arrowstyle": "<->", "lw": 1.5, "color": BLACK},
    )
    ax.text(0.50, 0.91, r"$\mathrm{OH+O}\;\rightleftharpoons\;\mathrm{H+O_2}$",
            ha="center", va="bottom", fontsize=12.0, color=BLACK)
    ax.plot(x_scaled, y_scaled, color=PURPLE, lw=3.0, solid_capstyle="round")
    ax.fill_between(x_scaled, 0.31, y_scaled, color=LIGHT_PURPLE, alpha=0.55)
    for label, mechanism, position in zip(
        ("M1", "M2"),
        ("low-gap\namplification", "slope\ncompensation"),
        maxima,
    ):
        xp, yp = x_scaled[position], y_scaled[position]
        ax.scatter([xp], [yp], marker="v", s=72, facecolor="white",
                   edgecolor=BLACK, linewidth=1.1, zorder=5)
        ax.text(xp, yp + 0.06, label, ha="center", va="bottom",
                fontsize=10.2, fontweight="bold", color=BLACK)
        ax.text(xp, 0.18, mechanism, ha="center", va="center",
                fontsize=9.1, color=BLACK, linespacing=1.0)
        ax.plot([xp, xp], [0.25, yp - 0.02], color=BLACK, lw=0.8,
                ls=(0, (3, 2)), zorder=0)
    ax.text(
        0.50, 0.025,
        r"$\alpha_0=A_{00}+2|M_{01}|^2/\Delta_{10}$",
        ha="center", va="bottom", fontsize=10.5, color=BLACK,
    )
    pdf = output_dir / "HO2_TOC_graphic.pdf"
    png = output_dir / "HO2_TOC_graphic.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pdf, png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="HO2_qgeom_protocol_validation_work")
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    output_dir = validation_root(workdir) / "analysis" / "manifold"
    rows = sorted(
        read_csv(output_dir / "manifold_summary.csv"),
        key=lambda row: int(row["source_index"]),
    )
    if not rows:
        raise SystemExit("Empty manifold_summary.csv")
    dataset = np.load(Path(__file__).resolve().parent / "data" / "01_irc.npz")
    source_count = len(dataset["R"])
    dataset.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10.2,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "text.color": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "savefig.transparent": False,
    })
    outputs = [
        *create_figure1(rows, source_count, output_dir),
        *create_toc(rows, source_count, output_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
