#!/usr/bin/env python3
"""Analyze global C2 rank-six geometry transport with nonorthogonal overlaps.

Unlike the local crossing analyzer, a successful global scan is not required
to create a new crossing bracket.  PASS means that every stored point and
every adjacent transport edge passes the continuity gates and that the
requested geometry window is covered.  All X/Delta sign changes are still
reported as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from pyscf import fci, gto
from pyscf.fci import addons as fci_addons
from pyscf.fci import cistring
from scipy.optimize import linear_sum_assignment

from c2_sa6_core import NCAS, NCORE, NELECAS, NSTATE, molecule


STATE_IRREPS = ("Ag", "Ag", "Ag", "B1g", "B2u", "B3u")
NRETAIN = NCORE + NCAS
FULL_NELECAS = (NELECAS[0] + NCORE, NELECAS[1] + NCORE)


def load_point(path: Path):
    with h5py.File(path, "r") as h5:
        point = {
            "path": path,
            "R": float(h5.attrs["R_angstrom"]),
            "converged": bool(h5.attrs.get("converged", False)),
            "energies": np.asarray(h5["e_states_Eh"], dtype=float),
            "mo": np.asarray(h5["mo_coeff"]),
            "ci": [np.asarray(x) for x in np.asarray(h5["ci_vectors"])],
            "spin_square": (
                np.asarray(h5["state_spin_square"], dtype=float)
                if "state_spin_square" in h5
                else np.zeros(NSTATE)
            ),
            "mu": (
                np.asarray(h5["mu_P_xyz"], dtype=float)
                if "mu_P_xyz" in h5
                else np.full((3, NSTATE, NSTATE), np.nan)
            ),
        }
        if "state_irreps_json" in h5.attrs:
            point["irreps"] = tuple(json.loads(h5.attrs["state_irreps_json"]))
        else:
            point["irreps"] = STATE_IRREPS
    if len(point["ci"]) != NSTATE or point["energies"].size != NSTATE:
        raise ValueError(f"Incomplete rank-six point: {path}")
    return point


def embed_active_ci(ci):
    active_strings = cistring.make_strings(range(NCAS), NELECAS[0])
    full_strings = cistring.make_strings(range(NRETAIN), FULL_NELECAS[0])
    full_address = {int(bitstring): i for i, bitstring in enumerate(full_strings)}
    addresses = [
        full_address[(int(active) << NCORE) | ((1 << NCORE) - 1)]
        for active in active_strings
    ]
    result = np.zeros((len(full_strings), len(full_strings)), dtype=np.asarray(ci).dtype)
    result[np.ix_(addresses, addresses)] = np.asarray(ci).reshape(
        len(active_strings), len(active_strings)
    )
    return result


def cross_ao_overlap(left_mol, right_mol):
    return gto.intor_cross("int1e_ovlp", left_mol, right_mol)


def state_overlap(left, right):
    left_mol = molecule(left["R"], symmetry=True)
    right_mol = molecule(right["R"], symmetry=True)
    s_ao = cross_ao_overlap(left_mol, right_mol)
    left_retained = left["mo"][:, :NRETAIN]
    right_retained = right["mo"][:, :NRETAIN]
    retained_overlap = left_retained.T @ s_ao @ right_retained
    active_overlap = (
        left["mo"][:, NCORE:NRETAIN].T
        @ s_ao
        @ right["mo"][:, NCORE:NRETAIN]
    )

    left_full = [embed_active_ci(ci) for ci in left["ci"]]
    right_full = [embed_active_ci(ci) for ci in right["ci"]]
    transformed_left = [
        fci_addons.transform_ci_for_orbital_rotation(
            ci, NRETAIN, FULL_NELECAS, retained_overlap
        )
        for ci in left_full
    ]
    overlap = np.empty((NSTATE, NSTATE))
    for i, bra in enumerate(transformed_left):
        for j, ket in enumerate(right_full):
            overlap[i, j] = np.dot(bra.ravel().conj(), ket.ravel()).real
    return overlap, active_overlap, retained_overlap


def anchor_identities(point):
    energies = point["energies"]
    ag = [i for i, irrep in enumerate(point["irreps"]) if irrep == "Ag"]
    b1g = next(i for i, irrep in enumerate(point["irreps"]) if irrep == "B1g")
    delta_ag = min(ag, key=lambda i: abs(energies[i] - energies[b1g]))
    sigma = sorted((i for i in ag if i != delta_ag), key=lambda i: energies[i])
    if len(sigma) != 2:
        raise RuntimeError("Cannot construct center Ag identities")
    identities = {
        delta_ag: "Delta_g_x2-y2",
        b1g: "Delta_g_xy",
        sigma[0]: "X_1Sigma_g+",
        sigma[1]: "Bp_1Sigma_g+",
    }
    for i, irrep in enumerate(point["irreps"]):
        if irrep == "B2u":
            identities[i] = "A_1Pi_u_y"
        elif irrep == "B3u":
            identities[i] = "A_1Pi_u_x"
    if len(identities) != NSTATE:
        raise RuntimeError(f"Incomplete center identities: {identities}")
    return identities


def assign_edge(left, right, left_identities):
    overlap, active_overlap, retained_overlap = state_overlap(left, right)
    polar_left, singular_values, polar_right_h = np.linalg.svd(overlap)
    polar_transport = polar_left @ polar_right_h
    polar_unitarity_error = float(
        np.max(
            np.abs(
                polar_transport.conj().T @ polar_transport - np.eye(NSTATE)
            )
        )
    )
    rows, cols = linear_sum_assignment(-np.abs(overlap))
    right_identities = {int(j): left_identities[int(i)] for i, j in zip(rows, cols)}
    matched = [abs(overlap[i, j]) for i, j in zip(rows, cols)]
    return right_identities, {
        "from_R_angstrom": left["R"],
        "to_R_angstrom": right["R"],
        "delta_R_angstrom": right["R"] - left["R"],
        "active_min_sv": float(np.linalg.svd(active_overlap, compute_uv=False).min()),
        "retained_min_sv": float(
            np.linalg.svd(retained_overlap, compute_uv=False).min()
        ),
        "state_projector_singular_values": singular_values.tolist(),
        "state_projector_min_sv": float(singular_values.min()),
        "polar_transport_unitarity_error": polar_unitarity_error,
        "minimum_matched_abs_state_overlap": float(min(matched)),
        "assignment_prev_to_current": [
            {
                "previous_solver_order": int(i),
                "current_solver_order": int(j),
                "identity": left_identities[int(i)],
                "overlap": float(overlap[i, j]),
            }
            for i, j in zip(rows, cols)
        ],
        "overlap_matrix": overlap,
        "polar_transport_matrix": polar_transport,
    }


def propagate(points, anchor_r):
    anchor_index = min(range(len(points)), key=lambda i: abs(points[i]["R"] - anchor_r))
    if abs(points[anchor_index]["R"] - anchor_r) > 1.0e-8:
        raise ValueError(f"No point at anchor R={anchor_r:.8f}")
    identities = {anchor_index: anchor_identities(points[anchor_index])}
    cumulative_transport = {anchor_index: np.eye(NSTATE)}
    edges = []

    for i in range(anchor_index + 1, len(points)):
        identities[i], edge = assign_edge(points[i - 1], points[i], identities[i - 1])
        cumulative_transport[i] = (
            cumulative_transport[i - 1] @ edge["polar_transport_matrix"]
        )
        edges.append(edge)
    for i in range(anchor_index - 1, -1, -1):
        identities[i], edge = assign_edge(points[i + 1], points[i], identities[i + 1])
        cumulative_transport[i] = (
            cumulative_transport[i + 1] @ edge["polar_transport_matrix"]
        )
        edges.append(edge)
    return identities, edges, cumulative_transport


def state_index(identity_map, name):
    return next(index for index, identity in identity_map.items() if identity == name)


def crossing_brackets(point_rows):
    rows = sorted(point_rows, key=lambda row: row["R_angstrom"])
    brackets = []
    for left, right in zip(rows[:-1], rows[1:]):
        gl = left["X_minus_Delta_gap_Eh"]
        gr = right["X_minus_Delta_gap_Eh"]
        if gl == 0.0 or gr == 0.0 or gl * gr < 0.0:
            if gr == gl:
                estimate = 0.5 * (left["R_angstrom"] + right["R_angstrom"])
            else:
                estimate = left["R_angstrom"] - gl * (
                    right["R_angstrom"] - left["R_angstrom"]
                ) / (gr - gl)
            brackets.append(
                {
                    "R_low_angstrom": left["R_angstrom"],
                    "R_high_angstrom": right["R_angstrom"],
                    "gap_low_Eh": gl,
                    "gap_high_Eh": gr,
                    "linear_crossing_estimate_angstrom": estimate,
                    "width_angstrom": right["R_angstrom"] - left["R_angstrom"],
                }
            )
    return brackets


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args):
    points_dir = args.points.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(points_dir.glob("R_*/point.h5"))
    points = sorted((load_point(path) for path in paths), key=lambda p: p["R"])
    if not points:
        raise FileNotFoundError(f"No R_*/point.h5 files in {points_dir}")

    identities, edges, cumulative_transport = propagate(points, args.anchor_R)
    grid_roles = {}
    if args.grid_csv:
        with args.grid_csv.resolve().open() as handle:
            for row in csv.DictReader(handle):
                grid_roles[round(float(row["R_angstrom"]), 10)] = row
    state_rows = []
    point_rows = []
    for point_index, point in enumerate(points):
        identity_map = identities[point_index]
        for solver_order in range(NSTATE):
            state_rows.append(
                {
                    "R_angstrom": point["R"],
                    "solver_order": solver_order,
                    "irrep": point["irreps"][solver_order],
                    "identity": identity_map[solver_order],
                    "energy_Eh": float(point["energies"][solver_order]),
                    "spin_square": float(point["spin_square"][solver_order]),
                }
            )

        x = state_index(identity_map, "X_1Sigma_g+")
        bp = state_index(identity_map, "Bp_1Sigma_g+")
        da = state_index(identity_map, "Delta_g_x2-y2")
        db = state_index(identity_map, "Delta_g_xy")
        pay = state_index(identity_map, "A_1Pi_u_y")
        pax = state_index(identity_map, "A_1Pi_u_x")
        delta_energy = 0.5 * (point["energies"][da] + point["energies"][db])
        pi_energy = 0.5 * (point["energies"][pay] + point["energies"][pax])
        degenerate_indices = [x, da, db]
        if np.isfinite(point["mu"]).all():
            restricted = point["mu"][:, degenerate_indices][:, :, degenerate_indices]
            max_deg_mu = float(np.max(np.abs(restricted)))
        else:
            max_deg_mu = float("nan")
        point_rows.append(
            {
                "R_angstrom": point["R"],
                "grid_role": grid_roles.get(
                    round(point["R"], 10), {}
                ).get("grid_role", "unlisted"),
                "plot_use": grid_roles.get(
                    round(point["R"], 10), {}
                ).get("plot_use", "1"),
                "E_X_Eh": float(point["energies"][x]),
                "E_Bp_Eh": float(point["energies"][bp]),
                "E_Delta_mean_Eh": float(delta_energy),
                "E_Pi_mean_Eh": float(pi_energy),
                "X_minus_Delta_gap_Eh": float(point["energies"][x] - delta_energy),
                "Delta_component_split_Eh": float(
                    abs(point["energies"][da] - point["energies"][db])
                ),
                "Pi_component_split_Eh": float(
                    abs(point["energies"][pay] - point["energies"][pax])
                ),
                "max_abs_Pdeg_mu_Pdeg_au": max_deg_mu,
                "converged": point["converged"],
                "max_abs_spin_square": float(np.max(np.abs(point["spin_square"]))),
            }
        )

    edge_rows = []
    overlap_archive = {}
    polar_archive = {}
    for edge in edges:
        key = f"R_{edge['from_R_angstrom']:.6f}_to_{edge['to_R_angstrom']:.6f}"
        overlap_archive[key] = edge.pop("overlap_matrix")
        polar_archive[key] = edge.pop("polar_transport_matrix")
        edge_rows.append(
            {
                "from_R_angstrom": edge["from_R_angstrom"],
                "to_R_angstrom": edge["to_R_angstrom"],
                "delta_R_angstrom": edge["delta_R_angstrom"],
                "active_min_sv": edge["active_min_sv"],
                "retained_min_sv": edge["retained_min_sv"],
                "state_projector_min_sv": edge["state_projector_min_sv"],
                "polar_transport_unitarity_error": edge[
                    "polar_transport_unitarity_error"
                ],
                "minimum_matched_abs_state_overlap": edge[
                    "minimum_matched_abs_state_overlap"
                ],
                "assignment_json": json.dumps(edge["assignment_prev_to_current"]),
            }
        )
    np.savez_compressed(output / "adjacent_state_overlap_matrices.npz", **overlap_archive)
    np.savez_compressed(
        output / "adjacent_polar_transport_matrices.npz", **polar_archive
    )
    cumulative_archive = {
        f"anchor_to_R_{points[index]['R']:.6f}": matrix
        for index, matrix in cumulative_transport.items()
    }
    np.savez_compressed(
        output / "cumulative_polar_transport_from_anchor.npz",
        **cumulative_archive,
    )
    cumulative_unitarity_errors = {
        points[index]["R"]: float(
            np.max(np.abs(matrix.conj().T @ matrix - np.eye(NSTATE)))
        )
        for index, matrix in cumulative_transport.items()
    }

    brackets = crossing_brackets(point_rows)
    edge_gate = {
        "minimum_active_sv_ge_0p995": min(row["active_min_sv"] for row in edge_rows)
        >= 0.995,
        "minimum_retained_sv_ge_0p995": min(row["retained_min_sv"] for row in edge_rows)
        >= 0.995,
        "minimum_state_projector_sv_ge_0p98": min(
            row["state_projector_min_sv"] for row in edge_rows
        )
        >= 0.98,
        "minimum_matched_overlap_ge_0p90": min(
            row["minimum_matched_abs_state_overlap"] for row in edge_rows
        )
        >= 0.90,
        "adjacent_polar_unitarity_error_le_1e-10": max(
            row["polar_transport_unitarity_error"] for row in edge_rows
        )
        <= 1.0e-10,
        "cumulative_polar_unitarity_error_le_1e-9": max(
            cumulative_unitarity_errors.values()
        )
        <= 1.0e-9,
        "all_points_converged": all(row["converged"] for row in point_rows),
        "all_states_singlet_abs_S2_le_1e-6": max(
            row["max_abs_spin_square"] for row in point_rows
        )
        <= 1.0e-6,
        "Delta_splits_le_1e-7_Eh": max(
            row["Delta_component_split_Eh"] for row in point_rows
        )
        <= 1.0e-7,
        "Pi_splits_le_1e-7_Eh": max(
            row["Pi_component_split_Eh"] for row in point_rows
        )
        <= 1.0e-7,
    }
    r_min = min(point["R"] for point in points)
    r_max = max(point["R"] for point in points)
    coverage_gate = {
        "requested_R_min_covered": (
            args.expected_R_min is None or r_min <= args.expected_R_min + 1.0e-8
        ),
        "requested_R_max_covered": (
            args.expected_R_max is None or r_max >= args.expected_R_max - 1.0e-8
        ),
    }
    if not all(edge_gate.values()) or not all(coverage_gate.values()):
        status = "REVIEW"
    else:
        status = "PASS"
    summary = {
        "status": status,
        "method": "seeded bidirectional global rank-six SA6-CASSCF geometry transport",
        "number_of_points": len(points),
        "R_min_angstrom": r_min,
        "R_max_angstrom": r_max,
        "requested_R_min_angstrom": args.expected_R_min,
        "requested_R_max_angstrom": args.expected_R_max,
        "anchor_R_angstrom": args.anchor_R,
        "crossing_brackets": brackets,
        "number_of_X_Delta_crossing_brackets": len(brackets),
        "minimum_active_subspace_sv": min(row["active_min_sv"] for row in edge_rows),
        "minimum_retained_subspace_sv": min(
            row["retained_min_sv"] for row in edge_rows
        ),
        "minimum_state_projector_sv": min(
            row["state_projector_min_sv"] for row in edge_rows
        ),
        "minimum_matched_abs_state_overlap": min(
            row["minimum_matched_abs_state_overlap"] for row in edge_rows
        ),
        "maximum_adjacent_polar_unitarity_error": max(
            row["polar_transport_unitarity_error"] for row in edge_rows
        ),
        "maximum_cumulative_polar_unitarity_error": max(
            cumulative_unitarity_errors.values()
        ),
        "maximum_abs_Pdeg_mu_Pdeg_au": float(
            np.nanmax([row["max_abs_Pdeg_mu_Pdeg_au"] for row in point_rows])
        ),
        "gate": edge_gate,
        "coverage_gate": coverage_gate,
        "next_step": (
            "If PASS, prepare the global signed-field ladder at plot_use points. "
            "If REVIEW, inspect the first failing point or transport edge before "
            "any response calculation."
        ),
    }

    write_csv(
        output / "transport_points.csv",
        sorted(point_rows, key=lambda row: row["R_angstrom"]),
        tuple(point_rows[0].keys()),
    )
    write_csv(
        output / "state_identity.csv",
        sorted(state_rows, key=lambda row: (row["R_angstrom"], row["solver_order"])),
        tuple(state_rows[0].keys()),
    )
    write_csv(output / "transport_edges.csv", edge_rows, tuple(edge_rows[0].keys()))
    (output / "transport_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-R", type=float, default=1.70)
    parser.add_argument("--grid-csv", type=Path)
    parser.add_argument("--expected-R-min", type=float)
    parser.add_argument("--expected-R-max", type=float)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
