@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Package boundary, setup commands, code style, testing policy, CI workflow list, and PR
rules are in `AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Julia 1.13.0 (the CI pin; Linux,
CPU only; installed from the official tarball).

| Command | Status |
|---|---|
| `julia --project=. -e 'using Pkg; Pkg.instantiate()'` | verified |
| `julia --project=. -e 'using Pkg; Pkg.test()'` | verified (212 passed) |
| `julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".", overwrite=true, verbose=true)'` then `git diff --exit-code` | verified (no diff; this is the CI `lint-format` job) |
| `julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'` | not run (`codecov.yml` runs it; upload needs a secret) |
| `julia --project=benchmark benchmark/run_benchmarks.jl` | not run (no CI job runs it; `benchmark/Project.toml` needs BenchmarkTools resolved first) |

- The formatter command rewrites files in place. Run it, check `git diff`, and keep only intended changes.
- `Manifest.toml` is gitignored, so `Pkg.test()` resolves fresh each time; there is no committed lockfile.
- CI (`ci.yml`) runs only Julia `1.13.0` on `ubuntu-latest`.

## Architecture map

- `src/SpikeStream.jl`: module definition and exports; includes `spike_features.jl`.
- `src/spike_features.jl`: all feature functions (`spike_count`, `spike_density`, `isi_stats`, `detect_bursts`,
  `windowed_spike_features`, `normalized_feature_vector`).
- `test/runtests.jl`: single test file. `test/fixtures/spike_vectors.json`: frozen golden vectors (LIM-41).
- `benchmark/`: separate project (`Project.toml`) with `run_benchmarks.jl`.
- `docs/` holds only `logo.png` (used by the README). `CHANGELOG.md` has an `[Unreleased]` section.
- `REVIEW.md`: reviewer checklist. `.agents/setup` and `.cursor/` are agent-environment bootstrap, not build steps.

## Invariants and non-goals

- Do not reintroduce `compute_hurst`, `compute_hawkes`, or `compute_gbm_surprise`; they belong to `kinetic-signals`.
- No FFI today; `JSON` is a test-only dependency (`[extras]`/`[targets]`), and the only runtime dependency is `Statistics`.
- Output ranges are documented in `README.md`; do not change them without updating the docs (`AGENTS.md`).
- `Project.toml` pins `julia = "1.13"` in `[compat]`, matching the CI version. Keep it explicit.
- Every `.jl` file (`src/`, `test/`, `benchmark/`) starts with `# SPDX-License-Identifier: MIT OR Apache-2.0`. Keep it on new files.
- New exported functions need docstrings (`AGENTS.md`, `REVIEW.md`).
- Third-party GitHub Actions are pinned to commit SHAs; checkout steps set `persist-credentials: false`.

## Gotchas

- `Project.toml` has no publish `exclude`/`files` list, so adding this file needs no manifest change.
- `.claude/` is gitignored (along with `.agents/` and most of `.cursor/`), so keep shared agent guidance in this file or `AGENTS.md`.
- `README.md` is HTML-heavy (centered logo and badges) and `REVIEW.md` sets no markdown-only rule for it.
- Never run `Pkg.register` or any other publish/registry command.
- Edit only what the task needs.
