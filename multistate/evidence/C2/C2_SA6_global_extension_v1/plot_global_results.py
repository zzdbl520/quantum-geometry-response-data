#!/usr/bin/env python3
"""Make the publication global C2 PES/response-geometry figure.

The plotter reads only the certified SA6 transport and signed-field analysis
tables. PASS segments are drawn independently: FAIL, REVIEW, EXCLUDED, and
missing geometries are never interpolated. A frozen-orbital diagnostic can be
annotated, but its numerical response is never spliced into the relaxed-CASSCF
curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


EH_TO_EV = 27.211386245988
COLORS = {
    "X": "#111111",
    "Delta": "#C43C39",
    "Pi": "#3B6FB6",
    "Bp": "#D48A16",
    "perp": "#3B6FB6",
    "parallel": "#D48A16",
    "iso": "#268F83",
    "crossing": "#7B61A8",
    "diagnostic": "#A76412",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return float(value)
    raise KeyError(f"None of the columns {keys!r} is present")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.7,
            "axes.labelsize": 9.6,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_axis(ax: plt.Axes, panel: str) -> None:
    ax.tick_params(which="major", top=True, right=True, direction="in")
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.text(
        0.018,
        0.965,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
    )


def crossing_brackets(
    points: Sequence[dict[str, str]],
) -> list[dict[str, float]]:
    brackets: list[dict[str, float]] = []
    for left, right in zip(points[:-1], points[1:]):
        r0 = float(left["R_angstrom"])
        r1 = float(right["R_angstrom"])
        g0 = float_value(left, "X_minus_Delta_gap_Eh")
        g1 = float_value(right, "X_minus_Delta_gap_Eh")
        if g0 == 0.0:
            rc = r0
        elif g0 * g1 > 0.0:
            continue
        elif g1 == g0:
            rc = 0.5 * (r0 + r1)
        else:
            rc = r0 - g0 * (r1 - r0) / (g1 - g0)
        if brackets and abs(rc - brackets[-1]["R_crossing"]) < 1.0e-7:
            continue
        brackets.append(
            {
                "R_low": min(r0, r1),
                "R_high": max(r0, r1),
                "R_crossing": float(rc),
            }
        )
    return brackets


def add_crossing_guides(
    axes: Iterable[plt.Axes], brackets: Sequence[dict[str, float]]
) -> None:
    for ax in axes:
        for bracket in brackets:
            ax.axvspan(
                bracket["R_low"],
                bracket["R_high"],
                color=COLORS["crossing"],
                alpha=0.075,
                lw=0,
                zorder=0,
            )
            ax.axvline(
                bracket["R_crossing"],
                color=COLORS["crossing"],
                lw=0.95,
                ls=(0, (3, 2)),
                zorder=1,
            )


def identity(row: dict[str, str]) -> str:
    return row.get(
        "zero_field_ground_identity", row.get("ground_identity", "")
    )


def is_pass(row: dict[str, str]) -> bool:
    return row.get("status", "PASS").upper() == "PASS"


def pass_segments(
    rows: Sequence[dict[str, str]], *, split_identity: bool
) -> list[list[dict[str, str]]]:
    """Return contiguous certified segments without bridging missing rows."""
    r_all = np.asarray([float(row["R_angstrom"]) for row in rows], dtype=float)
    dr = np.diff(np.unique(r_all))
    ordinary = dr[(dr > 1.0e-8) & (dr <= 0.075)]
    max_gap = (
        max(0.075, 1.65 * float(np.median(ordinary)))
        if ordinary.size else 0.075
    )
    segments: list[list[dict[str, str]]] = []
    active: list[dict[str, str]] = []
    for row in rows:
        if not is_pass(row):
            if active:
                segments.append(active)
                active = []
            continue
        discontinuity = False
        if active:
            discontinuity = (
                float(row["R_angstrom"])
                - float(active[-1]["R_angstrom"])
                > max_gap
            )
            if split_identity:
                discontinuity = (
                    discontinuity or identity(row) != identity(active[-1])
                )
        if discontinuity:
            segments.append(active)
            active = []
        active.append(row)
    if active:
        segments.append(active)
    return segments


def plot_segments(
    ax: plt.Axes,
    segments: Sequence[Sequence[dict[str, str]]],
    keys: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    styles: Sequence[object] | None = None,
    markers: Sequence[str] | None = None,
    identity_colors: bool = False,
) -> None:
    styles = styles or ["-"] * len(keys)
    markers = markers or ["o"] * len(keys)
    colors = colors or [COLORS["X"]] * len(keys)
    for segment in segments:
        r = np.asarray([float(row["R_angstrom"]) for row in segment])
        for index, key in enumerate(keys):
            y = np.asarray([float_value(row, key) for row in segment])
            color = colors[index]
            if identity_colors:
                color = (
                    COLORS["Delta"]
                    if "Delta" in identity(segment[0])
                    else COLORS["X"]
                )
            ax.plot(
                r,
                y,
                color=color,
                lw=1.55,
                ls=styles[index],
                marker=markers[index],
                ms=3.2,
                markerfacecolor="white" if index else color,
                markeredgecolor=color,
                markeredgewidth=0.75,
                zorder=3,
            )


def save_outputs(
    fig: plt.Figure,
    prefix: Path,
    dpi: int,
    provenance: dict[str, object],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        path = prefix.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi)
        outputs.append(str(path))
    provenance["outputs"] = outputs
    prefix.with_name(prefix.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--field-work", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("C2_global_SA6")
    )
    parser.add_argument("--minimum-pass-geometries", type=int, default=20)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    configure_style()
    zero_root = args.zero_root.resolve()
    field_work = args.field_work.resolve()
    transport_csv = zero_root / "analysis" / "transport_points.csv"
    response_csv = field_work / "analysis" / "global_response_observables.csv"
    points = read_csv(transport_csv)
    points = [
        row for row in points
        if row.get("plot_use", "1") in ("1", "True", "true", "PASS", "")
    ]
    points.sort(key=lambda row: float(row["R_angstrom"]))
    response = read_csv(response_csv)
    response.sort(key=lambda row: float(row["R_angstrom"]))
    pass_count = sum(is_pass(row) for row in response)
    if not points:
        raise SystemExit("No certified zero-field transport points were found")
    if pass_count < args.minimum_pass_geometries and not args.allow_incomplete:
        raise SystemExit(
            f"Only {pass_count} PASS response geometries were found; "
            f"need at least {args.minimum_pass_geometries}. Finish the "
            "direction-adapted global production or use --allow-incomplete "
            "for a visibly gapped provisional audit plot."
        )

    brackets = crossing_brackets(points)
    if len(brackets) != 2:
        raise SystemExit(f"Expected two X/Delta crossings, found {len(brackets)}")

    diagnostic = None
    if args.diagnostic_root:
        diagnostic_path = (
            args.diagnostic_root.resolve() / "fixed_casci_summary.json"
        )
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("status") != "PASS":
            raise SystemExit("The frozen-orbital diagnostic is not PASS")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 6.15),
        sharex=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.25},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    add_crossing_guides(axes.ravel(), brackets)

    r_pes = np.asarray([float(row["R_angstrom"]) for row in points])
    pes_specs = (
        ("E_X_Eh", COLORS["X"], r"$X\,^1\Sigma_g^+$"),
        ("E_Delta_mean_Eh", COLORS["Delta"], r"$B\,^1\Delta_g$"),
        ("E_Pi_mean_Eh", COLORS["Pi"], r"$A\,^1\Pi_u$"),
        ("E_Bp_Eh", COLORS["Bp"], r"$B'\,^1\Sigma_g^+$"),
    )
    all_energies = [
        np.asarray([float_value(row, key) for row in points])
        for key, _, _ in pes_specs
    ]
    e_ref = min(float(np.min(values)) for values in all_energies)
    for (_, color, label), values in zip(pes_specs, all_energies):
        ax_a.plot(
            r_pes,
            (values - e_ref) * EH_TO_EV,
            color=color,
            lw=1.55,
            marker="o",
            ms=2.6,
            markevery=max(1, len(r_pes) // 35),
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=label,
            zorder=3,
        )
    ax_a.set_ylabel(r"Energy, $E-E_{\rm min}$ (eV)")
    ax_a.legend(
        loc="best", frameon=False, ncol=2, columnspacing=0.9,
        handlelength=1.9, handletextpad=0.45,
    )
    finish_axis(ax_a, "a")

    identity_segments = pass_segments(response, split_identity=True)
    plot_segments(
        ax_b,
        identity_segments,
        ("alpha_perpendicular_au", "alpha_parallel_au", "alpha_isotropic_au"),
        styles=("-", (0, (4, 2)), (0, (1.5, 1.5))),
        markers=("o", "s", "D"),
        identity_colors=True,
    )
    ax_b.set_ylabel(r"Ground-state polarizability (a.u.)")
    identity_handles = [
        Line2D([0], [0], color=COLORS["X"], lw=1.7,
               label=r"$X\,^1\Sigma_g^+$"),
        Line2D([0], [0], color=COLORS["Delta"], lw=1.7,
               label=r"$^1\Delta_g$"),
    ]
    component_handles = [
        Line2D([0], [0], color="0.35", lw=1.55, marker="o",
               label=r"$\alpha_\perp$"),
        Line2D([0], [0], color="0.35", lw=1.55, ls=(0, (4, 2)), marker="s",
               markerfacecolor="white", label=r"$\alpha_\parallel$"),
        Line2D([0], [0], color="0.35", lw=1.55, ls=(0, (1.5, 1.5)),
               marker="D", markerfacecolor="white", label=r"$\bar\alpha$"),
    ]
    ax_b.legend(
        handles=identity_handles + component_handles,
        loc="best", frameon=False, ncol=2, columnspacing=0.8,
        handlelength=1.9, handletextpad=0.4,
    )
    finish_axis(ax_b, "b")

    response_segments = pass_segments(response, split_identity=False)
    plot_segments(
        ax_c,
        response_segments,
        (
            "Tr_gFF_target_perpendicular_au",
            "Tr_gFF_target_parallel_au",
            "Tr_gFF_target_isotropic_au",
        ),
        colors=(COLORS["perp"], COLORS["parallel"], COLORS["iso"]),
        styles=("-", (0, (4, 2)), (0, (1.5, 1.5))),
        markers=("o", "s", "D"),
    )
    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"Rank-3 $\mathrm{Tr}\,g_{FF}^{(P)}$ (a.u.)")
    ax_c.legend(
        handles=[
            Line2D([0], [0], color=COLORS["perp"], marker="o", label=r"$\perp$"),
            Line2D([0], [0], color=COLORS["parallel"], ls=(0, (4, 2)),
                   marker="s", markerfacecolor="white", label=r"$\parallel$"),
            Line2D([0], [0], color=COLORS["iso"], ls=(0, (1.5, 1.5)),
                   marker="D", markerfacecolor="white", label="isotropic"),
        ],
        loc="best", frameon=False, ncol=3, columnspacing=0.7,
        handlelength=1.6, handletextpad=0.35,
    )
    finish_axis(ax_c, "c")

    delta_keys = (
        "Delta_geom_target_perpendicular_eV",
        "Delta_geom_target_parallel_eV",
        "Delta_geom_target_isotropic_eV",
    )
    # Older analysis tables used an upper-case unit suffix.
    if response and delta_keys[0] not in response[0]:
        delta_keys = tuple(key[:-2] + "eV" for key in delta_keys)
    plot_segments(
        ax_d,
        response_segments,
        delta_keys,
        colors=(COLORS["perp"], COLORS["parallel"], COLORS["iso"]),
        styles=("-", (0, (4, 2)), (0, (1.5, 1.5))),
        markers=("o", "s", "D"),
    )
    ax_d.set_ylabel(r"$\Delta_{\rm geom}^{(P)}$ (eV)")
    if diagnostic is not None:
        r_diag = float(diagnostic["R_angstrom"])
        ax_d.axvline(
            r_diag, color=COLORS["diagnostic"], lw=1.0,
            ls=(0, (1, 2)), zorder=2,
        )
        ax_d.text(
            0.98,
            0.94,
            "longitudinal orbital-branch\nbifurcation (CASCI audit: PASS)",
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            color=COLORS["diagnostic"],
            fontsize=7.2,
        )
    finish_axis(ax_d, "d")

    for ax in axes[1, :]:
        ax.set_xlabel(r"C--C distance, $R$ ($\mathrm{\AA}$)")

    status_counts: dict[str, int] = {}
    for row in response:
        status = row.get("status", "PASS").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
    provenance: dict[str, object] = {
        "status": (
            "PASS" if pass_count >= args.minimum_pass_geometries
            else "PROVISIONAL"
        ),
        "zero_root": str(zero_root),
        "field_work": str(field_work),
        "diagnostic_root": (
            str(args.diagnostic_root.resolve()) if args.diagnostic_root else None
        ),
        "legacy_spreadsheet_used": False,
        "transport_point_count": len(points),
        "response_status_counts": status_counts,
        "crossings": brackets,
        "plot_policy": (
            "Only contiguous PASS rows are joined; frozen-orbital diagnostic "
            "values are not inserted into relaxed-CASSCF response curves."
        ),
    }
    prefix = args.output_prefix.resolve()
    save_outputs(fig, prefix, args.dpi, provenance)
    plt.close(fig)
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
