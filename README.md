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
snap = spike_features(spike_times; t_start=0.0, t_end=0.1)
normalizer = SpikeFeatureNormalizer(; max_density=200.0, max_isi_cv=2.0)
vec = transform(normalizer, snap)
```

The same normalizer configuration is intended for batch snapshots and future
streaming snapshots (#30) without special-case logic. Defaults match the
historical `normalized_feature_vector` behavior (`count_mode=:presence`,
`max_isi_cv=2.0`, `clip=true`).

## Event Types

- `SpikeEvent(time, neuron_id)`: one spike from one unit. Time must be finite; events sort by time, then `neuron_id`.
- `SpikeTrain(neuron_id, times; sorted=false)`: sorted spike times of one unit. Duplicate timestamps are kept; non-finite times throw `ArgumentError`.
- `spike_times(train)`, `neuron_id(x)`: accessors. `spike_times(train)` returns the train's own sorted vector (no copy) and can be passed to every function above.
- `spike_events(train_or_trains)`, `spike_trains(events)`: convert between per-unit trains and a time-sorted event list.

Allocation: `SpikeTrain(id, times)` copies `times` once (and sorts the copy). `SpikeTrain{T,I}(id, v)` aliases a sorted `v::Vector{T}` with zero allocations and throws if `v` is unsorted. `spike_times` never allocates.

The existing functions still take plain `AbstractVector{<:Real}` timestamps; the types are optional.

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
| `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`, `windowed_spike_features`, `spike_features`, `SpikeFeatureNormalizer` / `transform`, `normalized_feature_vector`, `SpikeFeatureStream` / `snapshot` | **SpikeStream.jl** | Public API of this package |
| Hurst exponent | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_hurst` (removed) |
| Hawkes intensity | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_hawkes` (removed) |
| Surprise / geometric Brownian motion (GBM) | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Formerly transitional `compute_gbm_surprise` (removed) |
| Entropy | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Not part of SpikeStream.jl |
| Volatility | [kinetic-signals](https://github.com/rmems/kinetic-signals) | Not part of SpikeStream.jl |

### Integration

- No foreign-function interface (FFI) between SpikeStream.jl and kinetic-signals today.
- Spike fixtures for cross-package tests live in this package: [`test/fixtures/spike_vectors.json`](test/fixtures/spike_vectors.json) (LIM-41).
- Window note: with explicit `t_end`, `windowed_spike_features` uses half-open windows `[t_start, t_end)`; `spike_count` / `spike_density` use inclusive ends. When `t_end` is omitted, the last spike time is the implicit right edge; a final window whose end reaches that edge is treated as closed so the spike is included. Window starts use `t_start + k*step` (integer `k`) rather than accumulated `cur += step`. A computed window end within a few ulps of `t_end` snaps to `t_end` for membership and density; interior spikes strictly before an open window end stay in that window. Window `density` is `count` divided by each window's actual emitted span, so any window clamped to `t_end` (not only the last) reports density over a shorter span and can exceed `count / window_size`. Fixtures encode this.
- kinetic-signals `shared_vectors.json` is Rust-only (not consumed by SpikeStream.jl).

## Quick Start

```julia
using SpikeStream

times = [0.001, 0.005, 0.009, 0.040, 0.042, 0.044, 0.090]

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

### Events and trains

```julia
using SpikeStream

a = SpikeTrain(1, [0.030, 0.010, 0.020])   # unsorted input is sorted on copy
b = SpikeTrain(2, [0.015, 0.015])           # duplicate timestamps are kept

spike_times(a)                 # [0.010, 0.020, 0.030]
isi_stats(spike_times(a))      # existing vector APIs work on a train's times

events = spike_events([a, b])  # time-sorted SpikeEvent list across units
trains = spike_trains(events)  # back to one SpikeTrain per neuron_id
```

### Streaming

`SpikeFeatureStream` incrementally ingests `SpikeEvent`s and reports per-unit
`SpikeFeatureSnapshot`s over stepped windows, with memory bounded by the window
horizon rather than the input history.

```julia
stream = SpikeFeatureStream(window = 0.1, step = 0.01)  # sliding; step == window for tumbling

push!(stream, SpikeEvent(0.101, 17))   # true
push!(stream, SpikeEvent(0.104, 83))
push!(stream, SpikeEvent(0.109, 17))
push!(stream, SpikeEvent(0.105, 17))   # false — late/out-of-order, counted in stream.dropped
push!(stream, SpikeTrain(17, [0.110, 0.112]))  # returns number accepted

snap = snapshot(stream)   # active window containing the newest accepted event
# (t_start, t_end, window, neurons) — neurons sorted by id, each with a
# SpikeFeatureSnapshot of count/density/isi_mean/isi_cv/burst_count
```

- Windows are `[t0 + k*step, t0 + k*step + window)` on a grid anchored at
  `t_start` (or the first accepted event) — the same half-open interior
  semantics as `windowed_spike_features`. `snapshot` reports the active window
  containing the newest event; it may still be partial.
- Events must arrive in non-decreasing time order. A `SpikeEvent` with a
  time earlier than the newest accepted time is dropped (`push!` returns `false`,
  `stream.dropped` counts drops); equal timestamps are accepted.
- `snap.neurons` is sorted by `neuron_id` and each `features` field is a
  `SpikeFeatureSnapshot`, so `SpikeFeatureNormalizer` / `transform` apply
  unchanged. Only units with spikes inside the window appear.
- `push!` also accepts a `SpikeTrain` (bulk ingest, returns accepted count).
  Per-unit buffers are evicted behind the active window, so memory stays
  bounded by active units and in-window spike counts.

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
