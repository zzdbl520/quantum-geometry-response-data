#!/usr/bin/env python3
"""Dependency-light structural tests; no PySCF calculation is launched."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from analyze_protocol_validation import (
    ensure_extremum_schema,
    fit_family,
    mechanism_label,
)
from analyze_manifold_response import (
    mark_plateau_extrema,
    plateau_statistics,
    robust_adjacent_orderings,
)
from ho2_core import write_csv
from manifold_core import (
    aligned_hamiltonian,
    best_candidate_subset,
    covariant_pair_operators,
    effective_energy_operator,
    ground_state_response_decomposition,
    plus_minus_metric_trace,
    quantum_metric_operator,
)
from protocol_core import ci_roots_n, field_path, protocol_nstate, zero_path


def main() -> int:
    package = Path(__file__).resolve().parent
    data = np.load(package / "data" / "01_irc.npz", allow_pickle=True)
    assert data["R"].shape[0] == 58
    assert protocol_nstate("ss") == 1
    assert protocol_nstate("sa4") == 4
    one = np.ones((3, 2))
    assert len(ci_roots_n(one, 1)) == 1
    four = np.ones((4, 3, 2))
    assert len(ci_roots_n(four, 4)) == 4
    assert mechanism_label(-10.0, 3.0) == "metric-curvature-dominated"
    assert mechanism_label(3.0, -10.0) == "energy-scale-curvature-dominated"

    candidates = np.asarray([
        [0.01, 0.99, 0.02, 0.00],
        [0.00, 0.02, 0.01, 0.98],
    ])
    selected = best_candidate_subset(candidates, 2)
    assert selected["indices"] == [1, 3]
    assert selected["singular_values"].min() > 0.97

    angle = 0.37
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    energies = np.asarray([-1.2, -0.7])
    aligned, singular_values = aligned_hamiltonian(rotation, energies)
    assert np.allclose(aligned, rotation @ np.diag(energies) @ rotation.T)
    assert np.allclose(singular_values, 1.0)

    field = 1.0e-3
    h0 = np.diag([-1.0, -0.8])
    dipole_ref = np.asarray([[0.2, 0.03], [0.03, -0.1]])
    alpha_ref = np.asarray([[4.0, 0.4], [0.4, 2.0]])
    hm = h0 - field * dipole_ref - 0.5 * field**2 * alpha_ref
    hp = h0 + field * dipole_ref - 0.5 * field**2 * alpha_ref
    dipole, alpha = covariant_pair_operators(h0, hm, hp, field)
    assert np.allclose(dipole, dipole_ref, atol=1.0e-10)
    assert np.allclose(alpha, alpha_ref, atol=1.0e-9)

    decomposition = ground_state_response_decomposition(
        np.asarray([-1.0, -0.8]), dipole_ref, alpha_ref
    )
    internal_ref = 2.0 * abs(dipole_ref[0, 1]) ** 2 / 0.2
    assert abs(decomposition["external_ground_alpha"] - 4.0) < 1.0e-12
    assert abs(decomposition["internal_ground_alpha"] - internal_ref) < 1.0e-12
    assert abs(
        decomposition["reconstructed_ground_alpha"] - (4.0 + internal_ref)
    ) < 1.0e-12
    check_field = 2.0e-4
    check_minus = np.linalg.eigvalsh(
        h0 - check_field * dipole_ref - 0.5 * check_field**2 * alpha_ref
    )[0]
    check_plus = np.linalg.eigvalsh(
        h0 + check_field * dipole_ref - 0.5 * check_field**2 * alpha_ref
    )[0]
    finite_difference_alpha = -(
        check_plus + check_minus - 2.0 * h0[0, 0]
    ) / check_field**2
    assert abs(
        finite_difference_alpha
        - decomposition["reconstructed_ground_alpha"]
    ) < 1.0e-6

    metric_ref = np.diag([3.0, 1.0])
    root = np.diag(np.sqrt(1.0 - field**2 * np.diag(metric_ref)))
    metric = quantum_metric_operator(root, root, field)
    assert np.allclose(metric, metric_ref, atol=1.0e-9)
    pm = np.diag(np.sqrt(1.0 - 4.0 * field**2 * np.diag(metric_ref)))
    assert abs(plus_minus_metric_trace(pm, field) - np.trace(metric_ref)) < 1.0e-9

    delta_ref = np.asarray([[0.7, 0.08], [0.08, 0.4]])
    metric_sqrt = np.diag(np.sqrt(np.diag(metric_ref)))
    response_ref = 2.0 * metric_sqrt @ delta_ref @ metric_sqrt
    delta, information = effective_energy_operator(response_ref, metric_ref)
    assert information["metric_supported_rank"] == 2
    assert np.allclose(delta, delta_ref, atol=1.0e-12)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert "protocol_validation/ss/zero" in str(zero_path(root, "ss", 5))
        assert "F_p1.0e-03.h5" in str(field_path(root, 5, 1.0e-3))

        # Regression: an unbracketed first row and a bracketed later row must
        # still share one DictWriter schema.
        heterogeneous = [{"source_index": 4}, {"source_index": 5}]
        ensure_extremum_schema(heterogeneous)
        heterogeneous[1]["local_vertex_s_A"] = -0.3
        assert list(heterogeneous[0]) == list(heterogeneous[1])
        write_csv(root / "heterogeneous_summary.csv", heterogeneous)
        assert (root / "heterogeneous_summary.csv").is_file()

    fields = (2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3)
    pair = {}
    summary = {}
    for index, alpha in ((4, 10.0), (5, 20.0), (6, 12.0)):
        pair[index] = {
            field: {"alpha_parallel_au": alpha + field**2}
            for field in fields
        }
        summary[index] = {"alpha_parallel_au": alpha}
    common, curves, rows = fit_family(pair, summary)
    assert len(common) == 4
    assert len(curves) == 12
    assert any(
        row["source_index"] == 5 and row["is_local_maximum"] == 1
        for row in rows
    )

    plateau = plateau_statistics(
        [{"value": value} for value in (27.65, 27.80, 27.75, 27.76)],
        "value",
    )
    assert plateau["relative_range"] < 0.01
    plateau_rows = []
    for index, external, internal, lower, median, upper in (
        (4, 15.7, 0.4, 15.8, 16.1, 16.2),
        (5, 19.5, 8.3, 27.6, 27.75, 27.8),
        (6, 22.2, 0.7, 22.8, 22.94, 23.0),
    ):
        plateau_rows.append({
            "source_index": index,
            "s_cart_A": float(index),
            "internal_gap_zero_Eh": 0.0068 if index == 5 else 0.02,
            "plateau_external_ground_alpha_median_au": external,
            "plateau_internal_manifold_ground_alpha_median_au": internal,
            "plateau_reconstructed_ground_alpha_lower_au": lower,
            "plateau_reconstructed_ground_alpha_median_au": median,
            "plateau_reconstructed_ground_alpha_upper_au": upper,
        })
    extrema, status = mark_plateau_extrema(plateau_rows)
    assert status[5] == "ROBUST_MAXIMUM"
    assert status[4] == "UNBRACKETED"
    assert status[6] == "UNBRACKETED"
    assert extrema[0]["mechanism_class"] == "internal-manifold-generated"
    assert extrema[0]["conservative_peak_margin_au"] > 4.0
    orderings = robust_adjacent_orderings(plateau_rows)
    assert [item["direction"] for item in orderings] == [
        "INCREASING", "DECREASING",
    ]
    print("SELF_TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
