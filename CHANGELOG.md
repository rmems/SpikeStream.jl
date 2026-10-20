# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `windowed_spike_features` builds window starts as `t0 + k*step` from an integer `k`, so float-step drift no longer emits a sliver extra bin or mis-bins times such as `0.3`. Overlapping windows no longer count spikes past an explicit `t_end`, and implicit `t_end` still includes the last spike in every window that ends there. Non-integer `window_size/step` ends snap to `t_end` (and to the nominal edge) so a last spike is not dropped and a time sitting on a window end is not counted twice.
- `spike_density` uses the observation-window duration when both bounds are provided, so a single spike in `[0, 1]` has density `1`. If either bound is omitted, fewer than 2 spikes still return `0`.
- Non-finite spike times (`NaN`, `±Inf`) raise `ArgumentError` on all public APIs.
- `normalized_feature_vector` sorts the input once and reuses that order for count, density, ISI, and burst features.

### Changed

- Document package boundary vs kinetic-signals (LIM-214).
- Add frozen spike feature fixtures and range checks under `test/fixtures/spike_vectors.json` (LIM-41); JSON is a test-only dependency.
- Pin CI test matrix to Julia `1.13.0` only (dropped `min`/`pre` floating aliases) and narrow `Project.toml`'s `julia` compat bound to `"1.13"` to match what CI actually exercises (#34).
- Repository metadata (CI/coverage badges, sibling-repo links, changelog PR link) updated to reflect the current `rmems/SpikeStream.jl` ownership after the transfer back from `Limen-Neural` (#34).

### Removed

- `compute_hurst`, `compute_hawkes`, and `compute_gbm_surprise` transitional
  time-series proxy functions; these now belong to `kinetic-signals`
  ([#22](https://github.com/rmems/SpikeStream.jl/pull/22)).

## [0.1.0] - 2026-03-23

### Added

- Initial release with spike-stream feature extraction functions:
  `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`,
  `windowed_spike_features`, and `normalized_feature_vector`.
