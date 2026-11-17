# Quantum geometry and molecular response: reproducibility data

This repository hosts two related but separate studies. Their datasets, source
scripts, figures, and reproducibility statements should be cited independently.

## Studies

| Study | Directory | Scope |
| --- | --- | --- |
| **JCP — Multistate Quantum Geometry Distinguishes Physical Response from Electronic-State Switching** | [multistate/](multistate/) | C2, B2, and H + O2 / OH + O; multistate projectors and response |
| **Single-state — Quantum Geometry Governs Molecular Response Reversals** | [singlestate/](singlestate/) | H2, N2, Cr2, F2, OsO4, and dicopper–dioxygen response |

Each directory has its own README describing numerical entry points and limitations.

## Reproducibility and provenance

- [Multistate data and verification](multistate/README.md)
- [Single-state data and verification](singlestate/README.md)
- Legacy root paths `data/` and `response_data.zip` are **temporarily retained** to avoid breaking previously distributed URLs and scripts. Their content is duplicated by the matching paths under `singlestate/` and should not be treated as a separate dataset.
- Both studies primarily supply derived data and reproducibility workflows; do not interpret the presence of figures or verification reports as evidence that all underlying electronic-structure wavefunctions can be regenerated from scratch.
- Cite a versioned release or archived DOI once assigned. Do not infer a DOI from a GitHub directory URL.

## Suggested stable links

- JCP: https://github.com/zzdbl520/quantum-geometry-response-data/tree/main/multistate
- Single-state: https://github.com/zzdbl520/quantum-geometry-response-data/tree/main/singlestate
