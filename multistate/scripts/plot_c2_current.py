#!/usr/bin/env python3
"""Plot the certified global C2 multistate response result.

Only quantities whose own axis-specific signed-field audit passes are plotted
as response curves.  Frozen-orbital CASCI values are used exclusively as a
continuity diagnostic and are never spliced into relaxed-CASSCF response data.
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
    "X": "#171717",
    "Delta": "#C74440",
    "Pi": "#3D73B9",
    "Bp": "#D58A17",
    "perp": "#3976BA",
    "parallel": "#D58A17",
    "iso": "#2A9186",
    "metric": "#3976BA",
    "scale": "#C74440",
    "crossing": "#7B61A8",
    "external": "#5C666D",
    "branch": "#B36B16",
    "fixed": "#258C82",
    "gate": "#777777",
}

FIT_RELATIVE_MAX = 0.05
COMMON_FRAME_ZERO_OFFSET_MAX_EH = 5.0e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.6,
            "axes.labelsize": 9.5,
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
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_axis(
    ax: plt.Axes,
    panel: str,
    *,
    right_ticks: bool = True,
) -> None:
    ax.tick_params(which="major", top=True, right=right_ticks, direction="in")
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    # Align every bold panel label to the left of its y spine.  The panel-a
    # legend, rather than the panel label, receives the dedicated right shift.
    panel_x = -0.065
    ax.text(
        panel_x,
        1.025,
        f"({panel})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def crossing_brackets(points: Sequence[dict[str, str]]) -> list[dict[str, float]]:
    points = sorted(points, key=lambda row: f(row, "R_angstrom"))
    brackets: list[dict[str, float]] = []
    for left, right in zip(points[:-1], points[1:]):
        r0, r1 = f(left, "R_angstrom"), f(right, "R_angstrom")
        g0, g1 = f(left, "X_minus_Delta_gap_Eh"), f(right, "X_minus_Delta_gap_Eh")
        if g0 * g1 > 0:
            continue
        rc = r0 if g0 == 0 else r1 if g1 == 0 else r0 - g0 * (r1 - r0) / (g1 - g0)
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


def external_delta_pi_crossing(points: Sequence[dict[str, str]]) -> dict[str, float]:
    """Locate the low-R Delta/Pi crossing that closes the target/environment gap."""
    points = sorted(points, key=lambda row: f(row, "R_angstrom"))
    brackets: list[dict[str, float]] = []
    for left, right in zip(points[:-1], points[1:]):
        r0, r1 = f(left, "R_angstrom"), f(right, "R_angstrom")
        g0 = f(left, "E_Pi_mean_Eh") - f(left, "E_Delta_mean_Eh")
        g1 = f(right, "E_Pi_mean_Eh") - f(right, "E_Delta_mean_Eh")
        if g0 * g1 > 0:
            continue
        rc = r0 if g0 == 0 else r1 if g1 == 0 else r0 - g0 * (r1 - r0) / (g1 - g0)
        brackets.append(
            {
                "R_low": min(r0, r1),
                "R_high": max(r0, r1),
                "gap_low_Eh": g0,
                "gap_high_Eh": g1,
                "R_crossing": float(rc),
            }
        )
    low_r = [row for row in brackets if 1.4 < row["R_crossing"] < 1.65]
    if len(low_r) != 1:
        raise SystemExit(f"Expected one low-R Delta/Pi external crossing, found {len(low_r)}")
    return low_r[0]


def add_crossings(axes: Iterable[plt.Axes], crossings: Sequence[dict[str, float]]) -> None:
    for ax in axes:
        for crossing in crossings:
            ax.axvspan(
                crossing["R_low"],
                crossing["R_high"],
                color=COLORS["crossing"],
                alpha=0.065,
                lw=0,
                zorder=0,
            )
            ax.axvline(
                crossing["R_crossing"],
                color=COLORS["crossing"],
                lw=0.95,
                ls=(0, (3, 2)),
                zorder=1,
            )


def add_external_crossing(
    axes: Iterable[plt.Axes], crossing: dict[str, float]
) -> None:
    """Mark a P--Q crossing without implying an internal projector crossing."""
    for ax in axes:
        ax.axvline(
            crossing["R_crossing"],
            color=COLORS["external"],
            lw=0.95,
            ls=(0, (1.2, 1.4)),
            zorder=1,
        )


def target_alpha_matrix(row: dict[str, str]) -> np.ndarray:
    """Rebuild the rank-three response matrix stored in a signed-field row."""
    return np.asarray(
        [
            [f(row, "alpha_X"), f(row, "alpha_X_Delta_x2-y2"), f(row, "alpha_X_Delta_xy")],
            [f(row, "alpha_X_Delta_x2-y2"), f(row, "alpha_Delta_x2-y2"), f(row, "alpha_Delta_component_coupling")],
            [f(row, "alpha_X_Delta_xy"), f(row, "alpha_Delta_component_coupling"), f(row, "alpha_Delta_xy")],
        ]
    )


def fit_target_alpha_audit(
    pair_rows: Sequence[dict[str, str]],
) -> tuple[float, float]:
    """Reproduce the target-only even-Hamiltonian spread and stability gates."""
    rows = sorted(pair_rows, key=lambda row: f(row, "field_au"))
    fields = np.asarray([f(row, "field_au") for row in rows])
    # Pair tables store -2[H_even(F)-H(0)]/F^2.  Multiplication by -F^2/2
    # reconstructs an equivalent relative common-frame Hamiltonian; a free
    # intercept makes the omitted absolute H(0) immaterial to this audit.
    matrices = np.asarray(
        [-0.5 * target_alpha_matrix(row) * f(row, "field_au") ** 2 for row in rows]
    )
    fmax2 = float(np.max(fields) ** 2)
    x = fields**2 / fmax2
    flat = matrices.reshape(len(matrices), -1)
    coefficients = np.polynomial.polynomial.polyfit(x, flat, 2)
    alpha = (-2.0 * coefficients[1] / fmax2).reshape(3, 3)
    alpha = 0.5 * (alpha + alpha.T)
    fitted = np.polynomial.polynomial.polyval(x, coefficients).T.reshape(matrices.shape)
    response_scale = max(0.5 * float(np.linalg.norm(alpha)) * fmax2, 1.0e-14)
    spread = float(
        np.max(np.linalg.norm(matrices - fitted, axis=(1, 2))) / response_scale
    )

    linear = np.polynomial.polynomial.polyfit(x, flat, 1)
    alpha_linear = (-2.0 * linear[1] / fmax2).reshape(3, 3)
    alpha_linear = 0.5 * (alpha_linear + alpha_linear.T)
    omit_smallest = np.polynomial.polynomial.polyfit(x[1:], flat[1:], 2)
    alpha_omit = (-2.0 * omit_smallest[1] / fmax2).reshape(3, 3)
    alpha_omit = 0.5 * (alpha_omit + alpha_omit.T)
    stability = max(
        float(np.linalg.norm(alpha - alpha_linear)),
        float(np.linalg.norm(alpha - alpha_omit)),
    ) / max(float(np.linalg.norm(alpha)), 1.0)
    return spread, stability


def fit_target_metric_audit(
    pair_rows: Sequence[dict[str, str]],
) -> tuple[float, float]:
    """Reproduce the target-projector metric limit audit without rank-six data."""
    rows = sorted(pair_rows, key=lambda row: f(row, "field_au"))
    x = np.asarray([f(row, "field_au") ** 2 for row in rows])
    y = np.asarray([f(row, "target_projector_metric_trace") for row in rows])
    all_fit = np.polyfit(x, y, 1)
    fitted = np.polyval(all_fit, x)
    small_fit = np.polyfit(x[:3], y[:3], 1)
    spread = float(np.max(np.abs(y - fitted)) / max(float(np.max(np.abs(y))), 1.0))
    stability = abs(float(all_fit[1] - small_fit[1])) / max(abs(float(all_fit[1])), 1.0)
    return spread, stability


def certify_target_rows(
    limit_rows: Sequence[dict[str, str]], pair_rows: Sequence[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Certify panel-c observables only against gates that act on rank three."""
    grouped: dict[tuple[float, str], list[dict[str, str]]] = {}
    for row in pair_rows:
        grouped.setdefault((f(row, "R_angstrom"), row["axis"]), []).append(row)
    certified: list[dict[str, str]] = []
    audit: list[dict[str, object]] = []
    for source in limit_rows:
        row = dict(source)
        key = (f(row, "R_angstrom"), row["axis"])
        group = grouped.get(key, [])
        if len(group) != 4:
            raise SystemExit(f"Expected four signed-field pairs for R={key[0]} axis={key[1]}")
        alpha_spread, alpha_stability = fit_target_alpha_audit(group)
        metric_spread, metric_stability = fit_target_metric_audit(group)
        checks = {
            "all_signed_pairs_pass": all(pair.get("status") == "PASS" for pair in group),
            "target_alpha_spread": alpha_spread <= FIT_RELATIVE_MAX,
            "target_alpha_stability": alpha_stability <= FIT_RELATIVE_MAX,
            "target_metric_spread": metric_spread <= FIT_RELATIVE_MAX,
            "target_metric_stability": metric_stability <= FIT_RELATIVE_MAX,
            "target_zero_offset": f(row, "common_frame_zero_offset_abs_max_Eh")
            <= COMMON_FRAME_ZERO_OFFSET_MAX_EH,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rank3_status = "PASS" if not failed else "FAIL"
        row["aggregate_status"] = row.get("status", "")
        row["status"] = rank3_status
        certified.append(row)
        audit.append(
            {
                "R_angstrom": key[0],
                "axis": key[1],
                "aggregate_status": source.get("status", ""),
                "rank3_component_status": rank3_status,
                "all_signed_pairs_pass": checks["all_signed_pairs_pass"],
                "target_alpha_spread": alpha_spread,
                "target_alpha_stability": alpha_stability,
                "target_metric_spread": metric_spread,
                "target_metric_stability": metric_stability,
                "target_zero_offset_abs_Eh": f(row, "common_frame_zero_offset_abs_max_Eh"),
                "failed_rank3_gates": " ".join(failed),
            }
        )
    return certified, audit


def identity(row: dict[str, str]) -> str:
    return row.get("zero_field_ground_identity", "")


def pass_segments(
    rows: Sequence[dict[str, str]],
    *,
    split_identity: bool = True,
    max_gap: float = 0.075,
) -> list[list[dict[str, str]]]:
    rows = sorted(rows, key=lambda row: f(row, "R_angstrom"))
    segments: list[list[dict[str, str]]] = []
    active: list[dict[str, str]] = []
    for row in rows:
        if row.get("status", "PASS").upper() != "PASS":
            if active:
                segments.append(active)
                active = []
            continue
        if active:
            gap = f(row, "R_angstrom") - f(active[-1], "R_angstrom") > max_gap
            branch = split_identity and identity(row) != identity(active[-1])
            if gap or branch:
                segments.append(active)
                active = []
        active.append(row)
    if active:
        segments.append(active)
    return segments


def plot_segmented(
    ax: plt.Axes,
    rows: Sequence[dict[str, str]],
    key: str,
    *,
    color: str,
    marker: str,
    linestyle: object = "-",
    label: str | None = None,
    open_marker: bool = False,
    split_identity: bool = True,
    zorder: int = 3,
) -> None:
    first = True
    for segment in pass_segments(rows, split_identity=split_identity):
        x = np.asarray([f(row, "R_angstrom") for row in segment])
        y = np.asarray([f(row, key) for row in segment])
        ax.plot(
            x,
            y,
            color=color,
            lw=1.55,
            ls=linestyle,
            marker=marker,
            ms=3.0,
            markerfacecolor="white" if open_marker else color,
            markeredgecolor=color,
            markeredgewidth=0.75,
            label=label if first else None,
            zorder=zorder,
        )
        first = False


def make_isotropic_rows(
    x_rows: Sequence[dict[str, str]], z_rows: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    x_map = {round(f(row, "R_angstrom"), 10): row for row in x_rows}
    z_map = {round(f(row, "R_angstrom"), 10): row for row in z_rows}
    output: list[dict[str, str]] = []
    for key in sorted(set(x_map) & set(z_map)):
        xr, zr = x_map[key], z_map[key]
        status = (
            "PASS"
            if xr.get("status") == "PASS" and zr.get("status") == "PASS"
            else "FAIL"
        )
        alpha = (
            2.0 * f(xr, "physical_ground_quadratic_response_F0")
            + f(zr, "physical_ground_quadratic_response_F0")
        ) / 3.0
        output.append(
            {
                "R_angstrom": xr["R_angstrom"],
                "zero_field_ground_identity": xr["zero_field_ground_identity"],
                "alpha_isotropic_au": repr(alpha),
                "status": status,
            }
        )
    return output


def diagnostic_rows(
    pair_rows: Sequence[dict[str, str]],
) -> list[dict[str, float]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in pair_rows:
        if row.get("axis") != "z":
            continue
        r = f(row, "R_angstrom")
        if r < 2.8:
            continue
        grouped.setdefault(r, []).append(row)
    result = []
    for r in sorted(grouped):
        rows = grouped[r]
        result.append(
            {
                "R_angstrom": r,
                "zero_target_sv_min": min(f(row, "zero_target_sv_min") for row in rows),
                "plus_minus_target_sv_min": min(
                    f(row, "plus_minus_target_sv_min") for row in rows
                ),
            }
        )
    return result


def write_certified_audit(
    path: Path,
    *,
    transport_summary: dict,
    crossings: Sequence[dict[str, float]],
    external_crossing: dict[str, float],
    x_rows: Sequence[dict[str, str]],
    z_rows: Sequence[dict[str, str]],
    iso_rows: Sequence[dict[str, str]],
    c_x_rows: Sequence[dict[str, str]],
    r155_audit: dict[str, object],
    diagnostics: Sequence[dict],
    failed_longitudinal_pairs: int,
) -> None:
    """Write the compact numerical audit shipped beside the figure."""
    x155 = next(row for row in x_rows if abs(f(row, "R_angstrom") - 1.55) < 1.0e-10)
    z_pass = [row for row in z_rows if row.get("status") == "PASS"]
    iso_pass = [row for row in iso_rows if row.get("status") == "PASS"]
    metric_max = max(c_x_rows, key=lambda row: f(row, "target_projector_metric_trace_F0"))
    delta_min = min(c_x_rows, key=lambda row: f(row, "Delta_geom_target_eV"))
    rows: list[dict[str, object]] = [
        {
            "category": "zero_field",
            "quantity": "transport_points",
            "value": "119/119",
            "status": "PASS",
            "interpretation": "Complete state-tracked SA6 geometry transport from 1.0 to 3.2 Angstrom",
        },
        {
            "category": "zero_field",
            "quantity": "first_X_Delta_crossing_R_Angstrom",
            "value": crossings[0]["R_crossing"],
            "status": "PASS",
            "interpretation": "Internal crossing within the rank-three X plus Delta target projector",
        },
        {
            "category": "zero_field",
            "quantity": "Delta_Pi_external_crossing_R_Angstrom",
            "value": external_crossing["R_crossing"],
            "status": "PHYSICAL_P_Q_CLOSURE",
            "interpretation": "Delta target state crosses Pi environment state; target projector is not spectrally isolated",
        },
        {
            "category": "zero_field",
            "quantity": "second_X_Delta_crossing_R_Angstrom",
            "value": crossings[1]["R_crossing"],
            "status": "PASS",
            "interpretation": "Internal crossing within the rank-three X plus Delta target projector",
        },
        {
            "category": "zero_field",
            "quantity": "maximum_abs_Pdeg_mu_Pdeg_au",
            "value": transport_summary["maximum_abs_Pdeg_mu_Pdeg_au"],
            "status": "PASS",
            "interpretation": "No projected linear Stark coupling in either internal X/Delta degenerate manifold",
        },
        {
            "category": "response",
            "quantity": "transverse_x_aggregate_limits",
            "value": f"{sum(row.get('status') == 'PASS' for row in x_rows)}/{len(x_rows)}",
            "status": "AUXILIARY_RANK6_FAILURE_AT_R_1p55",
            "interpretation": "Aggregate gate combines target and auxiliary rank-six fit stability",
        },
        {
            "category": "geometry",
            "quantity": "transverse_rank3_component_limits",
            "value": f"{sum(row.get('status') == 'PASS' for row in c_x_rows)}/{len(c_x_rows)}",
            "status": "PASS",
            "interpretation": "Panel-c data are certified by target-only signed-pair, response, metric, and zero-offset gates",
        },
        {
            "category": "geometry",
            "quantity": "R_1p55_aggregate_max_fit_stability",
            "value": x155["max_fit_stability"],
            "status": "FAIL_AUXILIARY_RANK6",
            "interpretation": "Not applied to panel c because the maximum is contributed by the auxiliary rank-six response matrix",
        },
        {
            "category": "geometry",
            "quantity": "R_1p55_target_alpha_fit_stability",
            "value": r155_audit["target_alpha_stability"],
            "status": "PASS",
            "interpretation": "Rank-three response-matrix stability is below the 0.05 gate",
        },
        {
            "category": "geometry",
            "quantity": "R_1p55_target_metric_fit_stability",
            "value": r155_audit["target_metric_stability"],
            "status": "PASS",
            "interpretation": "Rank-three metric stability is below the 0.05 gate",
        },
        {
            "category": "geometry",
            "quantity": "R_1p55_target_zero_offset_abs_Eh",
            "value": r155_audit["target_zero_offset_abs_Eh"],
            "status": "PASS",
            "interpretation": "Rank-three common-frame intercept offset is below 5e-8 Eh",
        },
        {
            "category": "response",
            "quantity": "longitudinal_z_zero_field_limits",
            "value": f"{len(z_pass)}/{len(z_rows)}",
            "status": "PARTIAL_CERTIFICATION",
            "interpretation": "Only PASS rows are used; no interpolation through failures",
        },
        {
            "category": "response",
            "quantity": "failed_optimized_longitudinal_pairs",
            "value": f"{failed_longitudinal_pairs}/60 sampled pairs",
            "status": "DIAGNOSTIC_FAILURE",
            "interpretation": "Optimized CASSCF leaves the target manifold from R=2.9 to 3.2 Angstrom",
        },
        {
            "category": "response",
            "quantity": "sampled_alpha_parallel_max_au",
            "value": max(f(row, "physical_ground_quadratic_response_F0") for row in z_pass),
            "status": "PASS",
            "interpretation": "Maximum over certified longitudinal samples",
        },
        {
            "category": "response",
            "quantity": "sampled_alpha_isotropic_max_au",
            "value": max(f(row, "alpha_isotropic_au") for row in iso_pass),
            "status": "PASS",
            "interpretation": "Maximum where both transverse and longitudinal components pass",
        },
        {
            "category": "geometry",
            "quantity": "maximum_certified_Tr_gFF_perpendicular_au",
            "value": metric_max["target_projector_metric_trace_F0"],
            "status": "PASS",
            "interpretation": f"Observed at R={f(metric_max, 'R_angstrom'):.2f} Angstrom",
        },
        {
            "category": "geometry",
            "quantity": "minimum_certified_Delta_geom_perpendicular_eV",
            "value": delta_min["Delta_geom_target_eV"],
            "status": "PASS",
            "interpretation": f"Observed at R={f(delta_min, 'R_angstrom'):.2f} Angstrom",
        },
    ]
    for diagnostic in diagnostics:
        label = str(diagnostic["R_angstrom"]).replace(".", "p")
        rows.extend(
            [
                {
                    "category": "fixed_CASCI",
                    "quantity": f"R_{label}_min_zero_target_sv",
                    "value": diagnostic["minimum_zero_target_sv"],
                    "status": "PASS",
                    "interpretation": "Frozen-orbital target manifold remains continuous",
                },
                {
                    "category": "fixed_CASCI",
                    "quantity": f"R_{label}_min_plus_minus_target_sv",
                    "value": diagnostic["minimum_plus_minus_target_sv"],
                    "status": "PASS",
                    "interpretation": "Frozen-orbital signed-field pair remains continuous",
                },
            ]
        )
    rows.append(
        {
            "category": "diagnosis",
            "quantity": "certified_assignment",
            "value": "ORBITAL_OPTIMIZATION_BRANCH_BIFURCATION",
            "status": "PASS",
            "interpretation": "Optimized-CASSCF failure is not a rank-three CI-manifold discontinuity",
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("category", "quantity", "value", "status", "interpretation")
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--field-work", type=Path, required=True)
    parser.add_argument("--diagnostic-roots", type=Path, nargs=3, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    configure_style()
    zero_root = args.zero_root.resolve()
    field_work = args.field_work.resolve()
    transport_summary = read_json(zero_root / "analysis" / "transport_summary.json")
    if transport_summary.get("status") != "PASS":
        raise SystemExit("Zero-field transport audit is not PASS")

    points = read_csv(zero_root / "analysis" / "transport_points.csv")
    if len(points) != 119 or any(row.get("converged") != "True" for row in points):
        raise SystemExit("Expected 119 converged zero-field transport points")
    crossings = crossing_brackets(points)
    if len(crossings) != 2:
        raise SystemExit(f"Expected two X/Delta crossings, found {len(crossings)}")
    external_crossing = external_delta_pi_crossing(points)

    limits = read_csv(field_work / "analysis" / "zero_field_limits.csv")
    pairs = read_csv(field_work / "analysis" / "signed_field_pairs.csv")
    x_rows = [row for row in limits if row.get("axis") == "x"]
    z_rows = [row for row in limits if row.get("axis") == "z"]
    c_x_rows, target_component_audit = certify_target_rows(x_rows, pairs)
    iso_rows = make_isotropic_rows(x_rows, z_rows)
    if sum(row.get("status") == "PASS" for row in x_rows) != 53:
        raise SystemExit("Unexpected transverse PASS count")
    z_failed_pairs = [
        row
        for row in pairs
        if row.get("axis") == "z" and row.get("status") != "PASS"
    ]
    if len(z_failed_pairs) != 60:
        raise SystemExit("Expected 60 failed optimized-CASSCF longitudinal pairs")
    if sum(row.get("status") == "PASS" for row in c_x_rows) != 54:
        raise SystemExit("Expected all 54 transverse rank-three rows to pass component-specific gates")
    r155_audit = next(
        row
        for row in target_component_audit
        if abs(float(row["R_angstrom"]) - 1.55) < 1.0e-10
    )
    if not (
        r155_audit["aggregate_status"] == "FAIL"
        and r155_audit["rank3_component_status"] == "PASS"
    ):
        raise SystemExit("R=1.55 must be aggregate-FAIL but rank-three-component PASS")

    diagnostics = [read_json(root.resolve() / "fixed_casci_summary.json") for root in args.diagnostic_roots]
    diagnostics.sort(key=lambda row: float(row["R_angstrom"]))
    for row in diagnostics:
        if row.get("status") != "PASS" or row.get("diagnosis") != "ORBITAL_OPTIMIZATION_BRANCH_BIFURCATION":
            raise SystemExit("Frozen-orbital diagnostic does not match the expected historical status")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.35, 6.05),
        gridspec_kw={"hspace": 0.14, "wspace": 0.36},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    add_crossings((ax_a, ax_b, ax_c), crossings)
    add_external_crossing((ax_a, ax_c), external_crossing)

    # (a) Complete, certified zero-field potential-energy curves.
    points_sorted = sorted(points, key=lambda row: f(row, "R_angstrom"))
    r_all = np.asarray([f(row, "R_angstrom") for row in points_sorted])
    plot_mask = np.asarray([row.get("plot_use") == "1" for row in points_sorted])
    specs = (
        ("E_X_Eh", COLORS["X"], r"$X\,^1\Sigma_g^+$"),
        ("E_Delta_mean_Eh", COLORS["Delta"], r"$B\,^1\Delta_g$"),
        ("E_Pi_mean_Eh", COLORS["Pi"], r"$A\,^1\Pi_u$"),
        ("E_Bp_Eh", COLORS["Bp"], r"$B'\,^1\Sigma_g^+$"),
    )
    energy_arrays = [np.asarray([f(row, key) for row in points_sorted]) for key, _, _ in specs]
    e_ref = min(float(values.min()) for values in energy_arrays)
    for (_, color, label), values in zip(specs, energy_arrays):
        y = (values - e_ref) * EH_TO_EV
        ax_a.plot(r_all, y, color=color, lw=1.55, label=label, zorder=3)
        ax_a.plot(
            r_all[plot_mask],
            y[plot_mask],
            ls="none",
            marker="o",
            ms=2.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.7,
            zorder=4,
        )
    ax_a.set_ylabel(r"Energy, $E-E_{\min}$ (eV)")
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 1.00),
        frameon=False,
        ncol=2,
        columnspacing=0.85,
        handlelength=1.9,
        handletextpad=0.4,
    )
    finish_axis(ax_a, "a")

    # (b) Componentwise certified relaxed polarizability.
    plot_segmented(
        ax_b,
        x_rows,
        "physical_ground_quadratic_response_F0",
        color=COLORS["perp"],
        marker="o",
        label=r"$\alpha_\perp$",
    )
    plot_segmented(
        ax_b,
        z_rows,
        "physical_ground_quadratic_response_F0",
        color=COLORS["parallel"],
        marker="s",
        linestyle=(0, (4, 2)),
        label=r"$\alpha_\parallel$",
        open_marker=True,
    )
    plot_segmented(
        ax_b,
        iso_rows,
        "alpha_isotropic_au",
        color=COLORS["iso"],
        marker="D",
        linestyle=(0, (1.5, 1.5)),
        label=r"$\bar\alpha$",
        open_marker=True,
    )
    ax_b.axvspan(2.9, 3.2, color=COLORS["branch"], alpha=0.075, lw=0, zorder=0)
    ax_b.text(
        crossings[0]["R_crossing"] + 0.025,
        10.65,
        "ground-state\nidentity switch",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=COLORS["crossing"],
    )
    ax_b.text(
        0.580,
        0.680,
        "longitudinal\norbital-continuity\nfailure",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=6.9,
        color=COLORS["branch"],
    )
    ax_b.set_ylabel("Ground-state polarizability (a.u.)")
    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.80),
        frameon=False,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.4,
    )
    finish_axis(ax_b, "b")

    # (c) Certified transverse quantum-geometric factorization.
    c_metric_left = [
        row for row in c_x_rows if f(row, "R_angstrom") < external_crossing["R_crossing"]
    ]
    c_metric_right = [
        row for row in c_x_rows if f(row, "R_angstrom") > external_crossing["R_crossing"]
    ]
    plot_segmented(
        ax_c,
        c_metric_left,
        "target_projector_metric_trace_F0",
        color=COLORS["metric"],
        marker="o",
        label=r"$\mathrm{Tr}\,g_{FF,\perp}^{(P)}$",
        split_identity=False,
    )
    plot_segmented(
        ax_c,
        c_metric_right,
        "target_projector_metric_trace_F0",
        color=COLORS["metric"],
        marker="o",
        split_identity=False,
    )
    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"Rank-3 $\mathrm{Tr}\,g_{FF,\perp}^{(P)}$ (a.u.)", color=COLORS["metric"])
    ax_c.tick_params(axis="y", colors="black")
    ax_c2 = ax_c.twinx()
    plot_segmented(
        ax_c2,
        c_x_rows,
        "Delta_geom_target_eV",
        color=COLORS["scale"],
        marker="s",
        linestyle=(0, (4, 2)),
        label=r"$\Delta_{\rm geom,\perp}^{(P)}$",
        open_marker=True,
        split_identity=False,
    )
    ax_c.text(
        external_crossing["R_crossing"] + 0.018,
        0.13,
        r"$^1\Delta_g/^1\Pi_u$ $P$--$Q$ closure",
        transform=ax_c.get_xaxis_transform(),
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="black",
    )
    ax_c2.set_ylabel(
        r"$\Delta_{\rm geom,\perp}^{(P)}$ (eV)",
        color=COLORS["scale"],
        labelpad=7.0,
    )
    ax_c2.set_ylim(-0.25, 6.50)
    ax_c2.tick_params(
        axis="y",
        which="major",
        left=False,
        right=True,
        direction="in",
        colors="black",
    )
    ax_c2.tick_params(
        axis="x",
        which="both",
        top=False,
        bottom=False,
        labeltop=False,
        labelbottom=False,
    )
    ax_c2.minorticks_off()
    ax_c2.spines["left"].set_visible(False)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["bottom"].set_visible(False)
    ax_c2.spines["right"].set_color("black")
    ax_c2.spines["right"].set_linewidth(1.0)
    handles = [
        Line2D([0], [0], color=COLORS["metric"], marker="o", label=r"$\mathrm{Tr}\,g_{FF,\perp}^{(P)}$"),
        Line2D([0], [0], color=COLORS["scale"], ls=(0, (4, 2)), marker="s", markerfacecolor="white", label=r"$\Delta_{\rm geom,\perp}^{(P)}$"),
    ]
    ax_c.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.43, 0.84),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.35,
    )
    finish_axis(ax_c, "c", right_ticks=False)

    # (d) Optimized-CASSCF branch collapse versus fixed-orbital CI continuity.
    diag = diagnostic_rows(pairs)
    rd = np.asarray([row["R_angstrom"] for row in diag])
    zero_sv = np.asarray([row["zero_target_sv_min"] for row in diag])
    pm_sv = np.asarray([row["plus_minus_target_sv_min"] for row in diag])
    ax_d.plot(
        rd,
        zero_sv,
        color=COLORS["Delta"],
        lw=1.45,
        marker="o",
        ms=3.2,
        label=r"optimized: $P_0\leftrightarrow P_F$",
        zorder=3,
    )
    ax_d.plot(
        rd,
        pm_sv,
        color=COLORS["parallel"],
        lw=1.35,
        ls=(0, (4, 2)),
        marker="s",
        ms=3.2,
        markerfacecolor="white",
        label=r"optimized: $P_{-F}\leftrightarrow P_{+F}$",
        zorder=3,
    )
    rfix = np.asarray([float(row["R_angstrom"]) for row in diagnostics])
    zfix = np.asarray([float(row["minimum_zero_target_sv"]) for row in diagnostics])
    pfix = np.asarray([float(row["minimum_plus_minus_target_sv"]) for row in diagnostics])
    # The two fixed-CASCI diagnostics are numerically almost identical.  Apply
    # a small display-only horizontal offset so that neither marker hides the
    # other; the underlying R values retained in the data are unchanged.
    fixed_marker_offset_angstrom = 0.004
    ax_d.scatter(
        rfix - fixed_marker_offset_angstrom,
        zfix,
        s=58,
        marker="*",
        color=COLORS["fixed"],
        edgecolor=COLORS["fixed"],
        label=r"fixed CASCI: $P_0\leftrightarrow P_F$",
        zorder=6,
    )
    ax_d.scatter(
        rfix + fixed_marker_offset_angstrom,
        pfix,
        s=34,
        marker="D",
        facecolor="white",
        edgecolor=COLORS["fixed"],
        linewidth=1.0,
        label=r"fixed CASCI: $P_{-F}\leftrightarrow P_{+F}$",
        zorder=5,
    )
    ax_d.axhline(0.98, color=COLORS["gate"], lw=0.8, ls=(0, (1.5, 2)), zorder=1)
    ax_d.axhline(0.95, color=COLORS["gate"], lw=0.8, ls=(0, (1.5, 2)), zorder=1)
    ax_d.axvspan(2.9, 3.2, color=COLORS["branch"], alpha=0.075, lw=0, zorder=0)
    add_crossings((ax_d,), (crossings[1],))
    ax_d.set_xlim(2.82, 3.21)
    ax_d.set_ylim(0.0, 1.045)
    ax_d.set_ylabel(r"Target-subspace $\sigma_{\min}$")
    ax_d.legend(
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.35,
    )
    finish_axis(ax_d, "d")

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(0.99, 3.21)
    for ax in (ax_c, ax_d):
        ax.set_xlabel(r"C--C distance, $R$ ($\mathrm{\AA}$)")
    ax_a.tick_params(labelbottom=False)
    ax_b.tick_params(labelbottom=False)

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        path = prefix.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=args.dpi)
        outputs.append(str(path))
    plt.close(fig)

    audit_path = prefix.parent / "C2_certified_audit.csv"
    write_certified_audit(
        audit_path,
        transport_summary=transport_summary,
        crossings=crossings,
        external_crossing=external_crossing,
        x_rows=x_rows,
        z_rows=z_rows,
        iso_rows=iso_rows,
        c_x_rows=c_x_rows,
        r155_audit=r155_audit,
        diagnostics=diagnostics,
        failed_longitudinal_pairs=len(z_failed_pairs),
    )
    outputs.append(str(audit_path))

    x_pass = [row for row in x_rows if row.get("status") == "PASS"]
    z_pass = [row for row in z_rows if row.get("status") == "PASS"]
    provenance = {
        "status": "CERTIFIED_COMPONENTWISE",
        "method": "equal-weight symmetry-complete SA6-CASSCF(8,8)/aug-cc-pVTZ",
        "transport_status": transport_summary.get("status"),
        "transport_point_count": len(points),
        "crossings": crossings,
        "external_delta_pi_crossing": external_crossing,
        "axis_specific_response_audit": {
            "transverse_x": {"PASS": len(x_pass), "non_PASS": len(x_rows) - len(x_pass)},
            "longitudinal_z": {"PASS": len(z_pass), "non_PASS": len(z_rows) - len(z_pass)},
            "failed_optimized_longitudinal_pairs_R_2p9_to_3p2": len(z_failed_pairs),
            "transverse_rank3_component": {
                "PASS": sum(row.get("status") == "PASS" for row in c_x_rows),
                "non_PASS": sum(row.get("status") != "PASS" for row in c_x_rows),
            },
            "R_1p55_component_audit": r155_audit,
        },
        "fixed_orbital_diagnostics": [
            {
                "R_angstrom": row["R_angstrom"],
                "status": row["status"],
                "diagnosis": row["diagnosis"],
                "minimum_zero_target_sv": row["minimum_zero_target_sv"],
                "minimum_plus_minus_target_sv": row["minimum_plus_minus_target_sv"],
            }
            for row in diagnostics
        ],
        "plot_policy": [
            "The current figure describes an orbital-continuity failure; historical diagnosis codes are preserved as source metadata.",
            "Only axis-specific PASS rows are joined as relaxed-CASSCF response curves.",
            "Ground-state response curves do not bridge failed rows or electronic ground-state identity changes.",
            "Panel-c rows are certified only by gates acting on the plotted rank-three target response and metric; auxiliary rank-six stability is reported separately.",
            "At R=1.55 Angstrom the plotted rank-three quantities pass every target-specific gate although the aggregate rank-six audit fails.",
            "The geometry-weighted scale remains connected through the Delta/Pi target-environment crossing and changes sign; the metric is split at that singular P-Q closure.",
            "The panel-c P-Q closure annotation is black for full-page legibility; the external-crossing guide remains gray to distinguish it from the purple internal crossings.",
            "Joint-projector geometric curves remain connected through internal X/Delta crossings because both states belong to P.",
            "Frozen-orbital CASCI values are plotted only as subspace-continuity diagnostics.",
            "The two fixed-CASCI marker series are displaced horizontally by +/-0.004 Angstrom solely for visibility; their underlying R values are unchanged.",
            "Bold parenthesized panel labels are placed outside the axes; the panel-c secondary y axis alone owns the right-side ticks and spine.",
            "No frozen-orbital response value is inserted into relaxed-CASSCF polarizability or geometric-factor curves.",
        ],
        "outputs": outputs,
    }
    provenance_path = prefix.with_name(prefix.name + "_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
