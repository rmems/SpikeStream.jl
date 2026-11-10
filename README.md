<p align="center">
  <img src="docs/logo.png" width="220" alt="SpikeStream.jl">
</p>

<h1 align="center">SpikeStream.jl</h1>
<p align="center">Spike-stream feature extraction for spiking neural systems</p>
<p align="center">
  <a href="https://github.com/rmems/SpikeStream.jl/actions/workflows/ci.yml" rel="noopener"><img src="https://github.com/rmems/SpikeStream.jl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/rmems/SpikeStream.jl" rel="noopener"><img src="https://codecov.io/gh/rmems/SpikeStream.jl/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="LICENSE-MIT" rel="noopener"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="LICENSE-APACHE" rel="noopener"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

---

SpikeStream.jl is focused on feature extraction from spike-event streams.

## Core Features

- `spike_count(spike_times; t_start, t_end)`
- `spike_density(spike_times; t_start, t_end)`
- `isi_stats(spike_times)`
- `detect_bursts(spike_times; max_isi, min_spikes)`
- `windowed_spike_features(spike_times; window_size, step)`
- `spike_features(spike_times)` → raw [`SpikeFeatureSnapshot`](#feature-extraction-vs-normalization)
- `SpikeFeatureNormalizer` + `transform(normalizer, snapshot)` → explicit scaling
- `normalized_feature_vector(spike_times)` → compatibility helper (snapshot + default normalizer)

## Feature extraction vs normalization

Feature extraction returns physical/statistical quantities (`SpikeFeatureSnapshot`:
count, density, ISI mean/CV, burst count). Normalization is a separate, explicit
policy (`SpikeFeatureNormalizer`) so callers can choose scales for their model or
dataset without baking experiment-specific constants into extraction.

```julia
snap = spike_features(times; t_start=0.0, t_end=0.1)
normalizer = SpikeFeatureNormalizer(; max_density=200.0, max_isi_cv=2.0)
vec = transform(normalizer, snap)
```

The same normalizer configuration is intended for batch snapshots and future
streaming snapshots (#30) without special-case logic. Defaults match the
historical `normalized_feature_vector` behavior (`count_mode=:presence`,
`max_isi_cv=2.0`, `clip=true`).

## Output Ranges

- `spike_count` → integer `>= 0`
- `spike_density` → real `>= 0` (with an explicit positive `[t_start, t_end]`, this is `n / duration` even for 0 or 1 spike; a missing bound is inferred from the data and still returns `0` when `n < 2`)
- `isi_stats` → all fields non-negative
- `detect_bursts` → vector of index ranges (possibly empty)
- `windowed_spike_features`:
  - `count >= 0`
  - `density >= 0`
  - `isi_mean >= 0`
  - `isi_cv >= 0`
  - `burst_count >= 0`
- `SpikeFeatureSnapshot` fields match the corresponding raw feature ranges above
- `transform` / `normalized_feature_vector` → length-4 vector; with default `clip=true`, each component is in `[0, 1]`

## Package Boundary

SpikeStream.jl owns spike-stream **feature extraction** only. Kinetic / signal-dynamics APIs live in the Rust sibling [rmems/kinetic-signals](https://github.com/rmems/kinetic-signals).

| Area | Owner | Notes |
| --- | --- | --- |
| `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`, `windowed_spike_features`, `spike_features`, `SpikeFeatureNormalizer` / `transform`, `normalized_feature_vector` | **SpikeStream.jl** | Public API of this package |
| Hurst exponent | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_hurst` (removed) |
| Hawkes intensity | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_hawkes` (removed) |
| Surprise / geometric Brownian motion (GBM) | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_gbm_surprise` (removed) |
| Entropy | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Not part of SpikeStream.jl |
| Volatility | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Not part of SpikeStream.jl |

### Integration

- No foreign-function interface (FFI) between SpikeStream.jl and kinetic-signals today.
- Spike fixtures for cross-package tests live in this package: [`test/fixtures/spike_vectors.json`](test/fixtures/spike_vectors.json) (LIM-41).
- Window note: with explicit `t_end`, `windowed_spike_features` uses half-open windows `[t_start, t_end)`; `spike_count` / `spike_density` use inclusive ends. Fixtures encode this.
- kinetic-signals `shared_vectors.json` is Rust-only (not consumed by SpikeStream.jl).

## Quick Start

```julia
using SpikeStream

spike_times = [0.001, 0.005, 0.009, 0.040, 0.042, 0.044, 0.090]

count = spike_count(spike_times)
density = spike_density(spike_times; t_start=0.0, t_end=0.1)
stats = isi_stats(spike_times)
bursts = detect_bursts(spike_times; max_isi=0.004, min_spikes=3)
windows = windowed_spike_features(spike_times; window_size=0.03, step=0.03)
snap = spike_features(spike_times; t_start=0.0, t_end=0.1)
vec = transform(SpikeFeatureNormalizer(; max_density=200.0), snap)
# compatibility helper (same default policy):
# vec = normalized_feature_vector(spike_times; t_start=0.0, t_end=0.1, max_density=200.0)
```

## Installation

SpikeStream.jl is not yet registered in the Julia General registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/SpikeStream.jl")
```

## License

Licensed under either of:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
