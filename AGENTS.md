# AGENTS.md

<!-- version: 1.0.0 | updated: 2026-07-22 -->
<!-- identity: SpikeStream.jl coding agent guide -->

Instructions for AI coding agents working on SpikeStream.jl.

## Agent Identity

You are a Julia development assistant working on SpikeStream.jl,
a spike-stream feature extraction package for spiking neural systems.
Follow the conventions below for all code changes.

## Repository Context

- Issue tracker: Linear issue markers use the acronym LIM (Linear issue ID prefix).

- SpikeStream.jl is a Julia package for spike-stream feature extraction in spiking neural systems (SNNs).
- Package boundary — these functions belong to SpikeStream.jl:
  - `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`, `windowed_spike_features`, `normalized_feature_vector`
- Sibling repo: [rmems/kinetic-signals](https://github.com/rmems/kinetic-signals) (Rust) owns Hurst, Hawkes, surprise/geometric Brownian motion (GBM), entropy, and volatility.
- Removed from SpikeStream.jl (do not reintroduce): `compute_hurst`, `compute_hawkes`, `compute_gbm_surprise` — now kinetic-signals only.
- Integration: no foreign-function interface (FFI) today; spike fixtures live in `test/fixtures/spike_vectors.json` (LIM-41); kinetic-signals `shared_vectors.json` is Rust-only.
- License: dual MIT / Apache-2.0. SPDX headers required on all source files.

## Setup Commands

```bash
# Instantiate dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Format check (CI uses JuliaFormatter with default settings)
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".", overwrite=true, verbose=true)'
git diff --exit-code  # fails if formatting changed anything
```

## Code Style

- JuliaFormatter with default settings (no custom `.JuliaFormatter.toml`)
- SPDX license header at top of every `.jl` file: `# SPDX-License-Identifier: MIT OR Apache-2.0`
- Docstrings for all exported functions
- No comments unless the rationale is non-obvious
- Prefer editing existing files over creating new ones
- Keep `Project.toml` `[compat]` entries explicit: `julia = "1.13"`

## Testing

- Tests live in `test/runtests.jl`
- Frozen spike feature fixtures: `test/fixtures/spike_vectors.json` (LIM-41; deterministic outputs + range invariants)
- Fixture note: `detect_bursts` golden params (`max_isi`/`min_spikes`) apply only to the explicit `detect_bursts` check; `windowed_spike_features.burst_count` and `normalized_feature_vector` burst component use API defaults (`max_isi=0.02`, `min_spikes=3`).
- Run full suite before pushing: `julia --project=. -e 'using Pkg; Pkg.test()'`
- All tests must pass; coverage upload is handled by Codecov CI
- Output ranges are documented in README — do not change without updating docs

## CI Workflows

- `.github/workflows/ci.yml` — tests on Julia `1.13.0` (pinned) + format check
- `.github/workflows/codecov.yml` — code coverage reporting
- All third-party actions are pinned to commit SHAs (managed by Dependabot)
- `persist-credentials: false` on all checkout steps

## PR Instructions

- Branch naming: `<type>/<description>` (e.g., `chore/add-codecov-coverage`)
- Commit messages: `type(scope): description` (conventional commits)
- Run tests and format check before pushing
- Do not merge your own PRs
- If your PR contains substantial contributions from a generative AI tool, please disclose so
