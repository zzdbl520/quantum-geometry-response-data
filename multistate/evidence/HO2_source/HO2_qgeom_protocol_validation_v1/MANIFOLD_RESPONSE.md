# Gauge-invariant finite-field response of a retained manifold

At a fixed nuclear geometry, let the zero-field retained frame be
`Psi_0=(|psi_1(0)>,...,|psi_m(0)>)`.  Four SA4 field roots are calculated
independently at each signed field.  From the rectangular many-electron
overlap between `Psi_0` and the four candidates, the code selects the `m`
columns that maximize the minimum singular value.  No root label is used as
an electronic-state identity.

For a selected signed-field frame `Psi_F`, define

```text
S_0F = Psi_0^dagger Psi_F = U Sigma V^dagger,
R_0F = U V^dagger.
```

`R_0F` is the unitary polar transport.  The diagonal field-state Hamiltonian
is represented in the zero-field gauge as

```text
H_F^(0) = R_0F diag(E_1(F),...,E_m(F)) R_0F^dagger.
```

The central finite differences then give the covariant internal dipole and
polarizability operators,

```text
M_F = [H_+^(0) - H_-^(0)] / (2 F),
A_F = -[H_+^(0) + H_-^(0) - 2 H_0] / F^2.
```

Their eigenvalues, traces, and the spectral width of `M_F` are invariant under
unitary rotations of the retained zero-field frame.  The spectral width of
`M_F` diagnoses first-order response internal to the manifold; `A_F` is the
smooth second-order covariant response operator.

The non-Abelian external quantum-metric operator is estimated without fixing
the gauges of the signed-field frames,

```text
G_FF = [2 I - S_0- S_0-^dagger - S_0+ S_0+^dagger] / (2 F^2).
```

Its trace is independently checked against the minus/plus chordal distance,

```text
Tr(G_FF) = [m - ||Psi_-^dagger Psi_+||_F^2] / (4 F^2) + O(F^2).
```

All matrices are extrapolated evenly in `F^2`.  On the numerically supported
metric range, the effective response-energy operator is defined by the exact
congruence

```text
Delta_eff = (1/2) G_FF^(-1/2) A_F G_FF^(-1/2),
A_F = 2 G_FF^(1/2) Delta_eff G_FF^(1/2).
```

The scalar trace summary is reported only as an invariant diagnostic.  It does
not replace the operator spectrum in a low-gap crossing region.

For a nondegenerate lowest state, the physical quadratic response is rebuilt
from the external response operator and the dipole coupling retained inside
the manifold,

```text
alpha_0 = A_00 + 2 sum_(j>0) |M_0j|^2 / (E_j-E_0).
```

The analysis reports the two terms separately.  Consequently, a maximum that
is absent from `Tr(A)/m` but present in the reconstructed ground response is a
low-gap internal-manifold maximum, not an optimizer artifact.  If an internal
gap truly closes, the perturbative reconstruction is disabled and the
eigenvalues of the effective manifold Hamiltonian must instead be inspected
for a linear `|F|` cusp.

Validation is deliberately split into two levels.  The strict operator gate
tests every transported matrix element and the extrapolated effective-energy
operator.  The observable-mechanism gate instead uses invariant raw-field
quantities over one fixed common window,
`2.5e-4,5.0e-4,7.5e-4,1.0e-3` a.u.  For each observable it reports the median
and full field envelope.  A robust maximum requires the center's lower
envelope to exceed both neighboring upper envelopes.  This permits a physical
ground-response mechanism to be validated without claiming that every
off-diagonal element of the full operator has converged.

Version 1.2.1 additionally separates the global plateau audit from the local
reaction topology.  A target is `ROBUST_NOT_MAXIMUM` when a neighboring lower
envelope lies above its upper envelope.  `local_extrema_validation_gate` asks
that every requested peak or nonpeak decision be robust, while
`reaction_topology_gate` combines those local decisions with stable manifold
rank.  Thus a narrowly failed plateau at a nonextremal endpoint remains an
explicit control warning but cannot manufacture or erase a locally bracketed
maximum.  `robust_adjacent_orderings` records the conservative margins for
every neighboring pair.

The smaller v1.1.1 window
`6.25e-5,1.25e-4,1.875e-4,2.5e-4` a.u. exposed subtractive cancellation in the
second energy difference and is retained only as a numerical-resolution audit.
