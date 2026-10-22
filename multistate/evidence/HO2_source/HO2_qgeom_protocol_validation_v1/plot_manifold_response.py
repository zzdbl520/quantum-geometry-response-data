#!/usr/bin/env python3
"""Plot the local gauge-invariant HO2 manifold-response audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ho2_core import read_csv
from protocol_core import validation_root


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    analysis = validation_root(workdir) / "analysis" / "manifold"
    rows = read_csv(analysis / "manifold_summary.csv")
    if not rows:
        raise SystemExit("Empty manifold summary; run analyze_manifold_response.py")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    s = numeric(rows, "s_cart_A")
    plateau_available = "plateau_reconstructed_ground_alpha_median_au" in rows[0]
    trace_alpha = numeric(
        rows,
        "plateau_trace_alpha_per_state_median_au"
        if plateau_available else "trace_alpha_per_state_au",
    )
    alpha_min = numeric(rows, "principal_alpha_min_au")
    alpha_max = numeric(rows, "principal_alpha_max_au")
    external_ground = numeric(
        rows,
        "plateau_external_ground_alpha_median_au"
        if plateau_available else "external_ground_alpha_au",
    )
    reconstructed_ground = numeric(
        rows,
        "plateau_reconstructed_ground_alpha_median_au"
        if plateau_available else "reconstructed_ground_alpha_au",
    )
    ss_alpha = numeric(rows, "ss_ground_alpha_au")
    sa2_alpha = numeric(rows, "sa2_ground_alpha_au")
    gap = numeric(rows, "internal_gap_zero_Eh")
    ss_overlap = numeric(rows, "ss_local_minimum_overlap")
    metric = numeric(rows, "trace_metric_per_state_au")
    delta = numeric(rows, "trace_Delta_geom_Eh")
    dipole_width = numeric(rows, "internal_dipole_width_au")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.fill_between(s, alpha_min, alpha_max, color="#6a3d9a", alpha=0.16,
                    label="principal-response spectrum")
    ax.plot(s, trace_alpha, "o-", color="#6a3d9a", lw=1.8,
            label=r"rank-2 trace response")
    ax.plot(s, external_ground, "^-", color="#cab2d6", lw=1.3,
            label=r"external $A_{00}$")
    ax.plot(s, reconstructed_ground, "o-", color="#e31a1c", lw=2.0,
            label=r"$A_{00}+2|M_{01}|^2/\Delta_{10}$")
    if plateau_available:
        reconstructed_lower = numeric(
            rows, "plateau_reconstructed_ground_alpha_lower_au"
        )
        reconstructed_upper = numeric(
            rows, "plateau_reconstructed_ground_alpha_upper_au"
        )
        ax.fill_between(
            s, reconstructed_lower, reconstructed_upper,
            color="#e31a1c", alpha=0.12, linewidth=0.0,
            label="common-field envelope",
        )
    ss_mask = np.isfinite(ss_alpha)
    if np.any(ss_mask):
        ax.plot(s[ss_mask], ss_alpha[ss_mask], "s-", color="#1f78b4", lw=1.4,
                label="SS ground root")
    sa2_mask = np.isfinite(sa2_alpha)
    if np.any(sa2_mask):
        ax.plot(s[sa2_mask], sa2_alpha[sa2_mask], "--", color="0.35", lw=1.4,
                label="SA2 ground root")
    ax.set_ylabel(r"Response (a.u.)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(s, gap, "o-", color="#d95f02", lw=1.8)
    ax.axhline(2.5e-2, color="0.45", ls="--", lw=1.0)
    ax.set_ylabel(r"Internal gap ($E_{\mathrm{h}}$)", color="#d95f02")
    ax.tick_params(axis="y", colors="#d95f02")
    twin = ax.twinx()
    twin.plot(s, ss_overlap, "s-", color="#1b9e77", lw=1.4)
    twin.axhline(0.75, color="0.45", ls=":", lw=1.0)
    twin.set_ylabel("SS adjacent-state overlap", color="#1b9e77")
    twin.tick_params(axis="y", colors="#1b9e77", direction="in")

    ax = axes[1, 0]
    ax.plot(s, metric, "o-", color="#e66101", lw=1.8)
    ax.set_ylabel(r"$\mathrm{Tr}(G_{FF})/m$ (a.u.)", color="#e66101")
    ax.tick_params(axis="y", colors="#e66101")
    twin = ax.twinx()
    twin.plot(s, delta, "s-", color="#1b9e77", lw=1.4)
    twin.set_ylabel(r"$\Delta_{\mathrm{geom}}^{\mathrm{tr}}$ ($E_{\mathrm{h}}$)",
                    color="#1b9e77")
    twin.tick_params(axis="y", colors="#1b9e77", direction="in")

    ax = axes[1, 1]
    ax.plot(s, dipole_width, "o-", color="#7570b3", lw=1.8)
    ax.set_ylabel(r"Spectral width of $P\mu P$ (a.u.)")
    ax.set_yscale("log" if np.all(dipole_width > 0.0) else "linear")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.set_xlabel(r"IRC arc coordinate, $s$ ($\mathrm{\AA}$)")
        ax.text(-0.14, 1.05, label, transform=ax.transAxes,
                fontweight="bold", va="bottom")
        ax.minorticks_off()

    output_pdf = analysis / "HO2_manifold_response.pdf"
    output_png = analysis / "HO2_manifold_response.png"
    fig.savefig(output_pdf, dpi=300)
    fig.savefig(output_png, dpi=300)
    print(output_pdf)
    print(output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
