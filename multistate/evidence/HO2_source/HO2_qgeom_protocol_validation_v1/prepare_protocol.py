#!/usr/bin/env python3
"""Prepare an isolated HO2 protocol-validation work directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ho2_core import geometry_label, parse_indices, pass_h5, write_csv, write_json
from protocol_core import VERSION, metadata_path


MASSES = np.asarray([15.999, 15.999, 1.00784])
SYMBOLS = ("O", "O", "H")


def canonical_linear_geometry(coords: np.ndarray, charges: np.ndarray):
    origin = np.average(coords, axis=0, weights=charges)
    shifted = coords - origin
    axis = shifted[1] - shifted[0]
    axis /= np.linalg.norm(axis)
    x = shifted @ axis
    aligned = np.zeros((3, 3))
    aligned[:, 0] = x
    aligned[:, 0] -= np.average(aligned[:, 0], weights=charges)
    reconstructed = np.outer(x, axis)
    perpendicular = shifted - reconstructed
    return aligned, {
        "maximum_linearization_displacement_A": float(
            np.max(np.linalg.norm(perpendicular, axis=1))
        ),
        "rms_linearization_displacement_A": float(
            np.sqrt(np.mean(perpendicular**2))
        ),
    }


def main() -> int:
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=str(package / "data" / "01_irc.npz")
    )
    parser.add_argument("--pilot-workdir", required=True)
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument(
        "--indices", default="0:20",
        help="Inclusive source indices. Re-run with 0:30 only after validation.",
    )
    parser.add_argument("--seed-index", type=int, default=13)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    pilot = Path(args.pilot_workdir).resolve()
    workdir = Path(args.workdir).resolve()
    data = np.load(source, allow_pickle=True)
    indices = parse_indices(args.indices, len(data["R"]))
    if args.seed_index not in indices:
        raise SystemExit("--seed-index must be included in --indices")
    pilot_seed = pilot / "zero" / f"irc_{args.seed_index:03d}.h5"
    if not pass_h5(pilot_seed):
        raise SystemExit(f"Missing converged SA2 pilot seed: {pilot_seed}")

    existing_metadata = metadata_path(workdir)
    if existing_metadata.exists():
        old = json.loads(existing_metadata.read_text())
        invariants = {
            "pilot_workdir": str(pilot),
            "source_file": str(source),
            "seed_source_index": args.seed_index,
        }
        mismatches = {
            key: (old.get(key), value)
            for key, value in invariants.items()
            if old.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                "Refusing to change validation-workdir invariants: "
                + json.dumps(mismatches, sort_keys=True)
            )

    raw_rows = []
    maximum_displacement = 0.0
    for source_index in indices:
        n = int(np.asarray(data["N"])[source_index].ravel()[0])
        charges = np.asarray(data["Z"])[source_index, :n].astype(float)
        if n != 3 or list(charges.astype(int)) != [8, 8, 1]:
            raise RuntimeError(
                f"Unexpected reaction-1 geometry {source_index}: "
                f"N={n}, Z={charges.tolist()}"
            )
        coords = np.asarray(data["R"])[source_index, :n].astype(float)
        aligned, report = canonical_linear_geometry(coords, charges)
        maximum_displacement = max(
            maximum_displacement,
            report["maximum_linearization_displacement_A"],
        )
        distance = lambda i, j: float(np.linalg.norm(aligned[i] - aligned[j]))
        raw_rows.append({
            "source_index": source_index,
            "label": geometry_label(source_index),
            "energy_kcal_mol": float(np.asarray(data["E"])[source_index].ravel()[0]),
            "r_OO_A": distance(0, 1),
            "r_O0H_A": distance(0, 2),
            "r_O1H_A": distance(1, 2),
            "linearization_max_A": report["maximum_linearization_displacement_A"],
            "linearization_rms_A": report["rms_linearization_displacement_A"],
            "coords": aligned,
        })

    cumulative = [0.0]
    cumulative_mw = [0.0]
    for left, right in zip(raw_rows[:-1], raw_rows[1:]):
        delta = right["coords"] - left["coords"]
        cumulative.append(cumulative[-1] + float(np.linalg.norm(delta)))
        cumulative_mw.append(
            cumulative_mw[-1]
            + float(np.sqrt(np.sum(MASSES[:, None] * delta**2)))
        )
    seed_position = indices.index(args.seed_index)
    s0 = cumulative[seed_position]
    smw0 = cumulative_mw[seed_position]

    geometry_dir = workdir / "geometries"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, item in enumerate(raw_rows):
        xyz = geometry_dir / f"{item['label']}.xyz"
        lines = ["3", f"reaction 1 source index {item['source_index']}"]
        lines.extend(
            f"{symbol:2s} {coord[0]: .14f} {coord[1]: .14f} {coord[2]: .14f}"
            for symbol, coord in zip(SYMBOLS, item["coords"])
        )
        xyz.write_text("\n".join(lines) + "\n")
        rows.append({
            "grid_position": position,
            "source_index": item["source_index"],
            "label": item["label"],
            "is_seed": int(item["source_index"] == args.seed_index),
            "s_cart_A": cumulative[position] - s0,
            "s_mw_sqrtamu_A": cumulative_mw[position] - smw0,
            "energy_kcal_mol": item["energy_kcal_mol"],
            "r_OO_A": item["r_OO_A"],
            "r_O0H_A": item["r_O0H_A"],
            "r_O1H_A": item["r_O1H_A"],
            "linearization_max_A": item["linearization_max_A"],
            "coords_json": json.dumps(item["coords"].tolist()),
            "xyz_path": str(xyz),
        })
    write_csv(workdir / "geometry_grid.csv", rows)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "package_version": VERSION,
        "pilot_workdir": str(pilot),
        "source_file": str(source),
        "source_sha256": digest,
        "source_repository": "https://github.com/THGLab/H2COMBUSTION_DATA",
        "source_dataset_doi": "10.6084/m9.figshare.19601689",
        "reaction": "H + O2 <-> OH + O",
        "selected_source_indices": indices,
        "seed_source_index": args.seed_index,
        "maximum_linearization_displacement_A": maximum_displacement,
        "protocols": {
            "ss": "state-specific ground-state CASSCF(13,9)",
            "sa4": "equal-weight four-state CASSCF(13,9)",
        },
        "coordinate_convention": (
            "O-O defines +x; charge center is the origin; off-axis noise is "
            "projected out; s=0 at source index 13"
        ),
    }
    write_json(metadata_path(workdir), metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Geometry grid: {workdir / 'geometry_grid.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
