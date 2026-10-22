#!/usr/bin/env python3
"""Pure linear-algebra machinery for gauge-invariant manifold response."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def hermitize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    return 0.5 * (value + value.conj().T)


def best_candidate_subset(overlap: np.ndarray, rank: int):
    """Select candidate columns that best preserve an anchor subspace."""
    value = np.asarray(overlap)
    rank = int(rank)
    if value.ndim != 2 or value.shape[0] != rank:
        raise ValueError(
            f"Expected a ({rank}, ncandidate) overlap, got {value.shape}"
        )
    if value.shape[1] < rank:
        raise ValueError("Fewer candidate states than requested manifold rank")
    best = None
    for columns in itertools.combinations(range(value.shape[1]), rank):
        block = value[:, columns]
        singular_values = np.linalg.svd(block, compute_uv=False)
        score = (
            float(singular_values.min()),
            float(np.sum(singular_values**2)),
            -float(np.mean(columns)),
        )
        if best is None or score > best[0]:
            best = (score, list(columns), singular_values, block)
    assert best is not None
    return {
        "indices": best[1],
        "singular_values": np.asarray(best[2]),
        "overlap": np.asarray(best[3]),
    }


def polar_unitary(overlap: np.ndarray):
    """Nearest unitary polar factor carrying a field frame into an anchor frame."""
    value = np.asarray(overlap)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"Polar transport needs a square overlap, got {value.shape}")
    left, singular_values, right_h = np.linalg.svd(value)
    return left @ right_h, singular_values


def aligned_hamiltonian(overlap: np.ndarray, energies: Sequence[float]):
    """Represent a diagonal field Hamiltonian in the polar-aligned anchor gauge."""
    energy = np.asarray(energies, dtype=float)
    polar, singular_values = polar_unitary(overlap)
    if polar.shape != (len(energy), len(energy)):
        raise ValueError("Energy count does not match overlap dimension")
    value = polar @ np.diag(energy) @ polar.conj().T
    return hermitize(value), singular_values


def covariant_pair_operators(
    h_zero: np.ndarray,
    h_minus: np.ndarray,
    h_plus: np.ndarray,
    field: float,
):
    """Central finite-difference dipole and polarizability operators."""
    field = float(field)
    if field <= 0.0:
        raise ValueError("Field magnitude must be positive")
    h0 = np.asarray(h_zero)
    hm = np.asarray(h_minus)
    hp = np.asarray(h_plus)
    dipole = hermitize((hp - hm) / (2.0 * field))
    polarizability = hermitize(-(hp + hm - 2.0 * h0) / field**2)
    return dipole, polarizability


def _nonnegative_defect(value: float, *, tolerance: float = 1.0e-8) -> float:
    number = float(value)
    if number < -tolerance:
        raise ValueError(f"Subspace fidelity exceeds its rank by {-number:.3e}")
    return max(number, 0.0)


def quantum_metric_operator(
    zero_minus: np.ndarray,
    zero_plus: np.ndarray,
    field: float,
):
    """Symmetric finite-difference non-Abelian quantum metric in anchor gauge."""
    sm = np.asarray(zero_minus)
    sp = np.asarray(zero_plus)
    if sm.shape != sp.shape or sm.ndim != 2 or sm.shape[0] != sm.shape[1]:
        raise ValueError("Zero/field overlaps must be equally sized square matrices")
    identity = np.eye(sm.shape[0], dtype=np.result_type(sm, sp))
    metric = (
        2.0 * identity - sm @ sm.conj().T - sp @ sp.conj().T
    ) / (2.0 * float(field) ** 2)
    metric = hermitize(metric)
    eig, vec = np.linalg.eigh(metric)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    if float(eig.min()) < -1.0e-7 * scale:
        raise ValueError(
            f"Finite-difference metric is not positive semidefinite: "
            f"min eigenvalue={eig.min():.3e}"
        )
    eig = np.maximum(eig, 0.0)
    return hermitize((vec * eig) @ vec.conj().T)


def plus_minus_metric_trace(
    minus_plus: np.ndarray,
    field: float,
) -> float:
    value = np.asarray(minus_plus)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Minus/plus overlap must be square")
    defect = _nonnegative_defect(
        value.shape[0] - float(np.linalg.norm(value, ord="fro") ** 2)
    )
    return defect / (4.0 * float(field) ** 2)


def fit_even_scalar(fields: Sequence[float], values: Sequence[float]):
    field = np.asarray(fields, dtype=float)
    data = np.asarray(values, dtype=float)
    if len(field) != len(data) or len(field) < 3:
        raise ValueError("At least three matched field/value pairs are required")
    x = field**2
    degree = 2 if len(field) >= 4 else 1
    coeff = np.polynomial.polynomial.polyfit(x, data, degree)
    fitted = np.polynomial.polynomial.polyval(x, coeff)
    limit = float(coeff[0])
    scale = max(float(np.max(np.abs(data))), 1.0)
    residual = float(np.max(np.abs(data - fitted)) / scale)
    reduced_degree = min(degree, len(field) - 2)
    reduced = np.polynomial.polynomial.polyfit(
        x[:-1], data[:-1], reduced_degree
    )
    stability = abs(limit - float(reduced[0])) / max(abs(limit), 1.0)
    return limit, residual, stability


def fit_even_matrix(fields: Sequence[float], matrices: Sequence[np.ndarray]):
    values = np.asarray(matrices)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Expected a sequence of square matrices")
    rank = values.shape[1]
    limit = np.zeros((rank, rank), dtype=values.dtype)
    residual = 0.0
    stability = 0.0
    for i in range(rank):
        for j in range(rank):
            entry, entry_residual, entry_stability = fit_even_scalar(
                fields, np.real(values[:, i, j])
            )
            limit[i, j] = entry
            residual = max(residual, entry_residual)
            stability = max(stability, entry_stability)
    return hermitize(limit), float(residual), float(stability)


def effective_energy_operator(
    polarizability: np.ndarray,
    metric: np.ndarray,
    *,
    relative_floor: float = 1.0e-10,
):
    """Return Delta from A = 2 G^(1/2) Delta G^(1/2) by congruence."""
    response = hermitize(np.asarray(polarizability))
    geometry = hermitize(np.asarray(metric))
    eig, vec = np.linalg.eigh(geometry)
    largest = max(float(eig.max()), 0.0)
    floor = max(relative_floor * largest, 1.0e-14)
    retained = eig > floor
    if not np.any(retained):
        raise ValueError("Quantum metric has no numerically supported direction")
    inverse_sqrt = (
        vec[:, retained] / np.sqrt(eig[retained])
    ) @ vec[:, retained].conj().T
    delta = hermitize(0.5 * inverse_sqrt @ response @ inverse_sqrt)
    return delta, {
        "metric_eigenvalues": eig,
        "metric_supported_rank": int(np.count_nonzero(retained)),
        "metric_condition_number": float(eig[retained].max() / eig[retained].min()),
    }


def ground_state_response_decomposition(
    energies: Sequence[float],
    dipole: np.ndarray,
    polarizability: np.ndarray,
    *,
    gap_floor: float = 1.0e-12,
):
    """Split the lowest-state response into external and internal parts.

    In the zero-field energy eigenbasis, the retained-manifold effective
    Hamiltonian is ``H(F)=H(0)+F M-F**2 A/2+...``.  For a nondegenerate
    lowest state, perturbation theory gives

    ``alpha_0=A_00+2 sum_{j>0}|M_0j|**2/(E_j-E_0)``.

    A true internal degeneracy is reported as unsupported rather than
    converted into a spurious large quadratic response.
    """
    energy = np.asarray(energies, dtype=float)
    moment = hermitize(np.asarray(dipole))
    response = hermitize(np.asarray(polarizability))
    if energy.ndim != 1 or len(energy) < 1:
        raise ValueError("At least one zero-field energy is required")
    if moment.shape != (len(energy), len(energy)):
        raise ValueError("Dipole matrix dimension does not match the energies")
    if response.shape != moment.shape:
        raise ValueError("Polarizability and dipole matrices must have one shape")

    external = float(np.real(response[0, 0]))
    if len(energy) == 1:
        return {
            "external_ground_alpha": external,
            "internal_ground_alpha": 0.0,
            "reconstructed_ground_alpha": external,
            "minimum_internal_gap": float("inf"),
            "perturbative_internal_response_supported": True,
        }

    gaps = energy[1:] - energy[0]
    minimum_gap = float(np.min(gaps))
    supported = bool(np.all(gaps > float(gap_floor)))
    if supported:
        internal = float(2.0 * np.sum(np.abs(moment[0, 1:]) ** 2 / gaps))
        total = external + internal
    else:
        internal = float("nan")
        total = float("nan")
    return {
        "external_ground_alpha": external,
        "internal_ground_alpha": internal,
        "reconstructed_ground_alpha": total,
        "minimum_internal_gap": minimum_gap,
        "perturbative_internal_response_supported": supported,
    }


__all__ = [
    "aligned_hamiltonian",
    "best_candidate_subset",
    "covariant_pair_operators",
    "effective_energy_operator",
    "fit_even_matrix",
    "fit_even_scalar",
    "ground_state_response_decomposition",
    "hermitize",
    "plus_minus_metric_trace",
    "polar_unitary",
    "quantum_metric_operator",
]
