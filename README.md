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
- `normalized_feature_vector(spike_times)`

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
- `normalized_feature_vector` → length-4 vector in `[0, 1]`

## Package Boundary

SpikeStream.jl owns spike-stream **feature extraction** only. Kinetic / signal-dynamics APIs live in the Rust sibling [rmems/kinetic-signals](https://github.com/rmems/kinetic-signals).

| Area | Owner | Notes |
| --- | --- | --- |
| `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`, `windowed_spike_features`, `normalized_feature_vector` | **SpikeStream.jl** | Public API of this package |
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

times = [0.001, 0.005, 0.009, 0.040, 0.042, 0.044, 0.090]

count = spike_count(times)
density = spike_density(times; t_start=0.0, t_end=0.1)
stats = isi_stats(times)
bursts = detect_bursts(times; max_isi=0.004, min_spikes=3)
windows = windowed_spike_features(times; window_size=0.03, step=0.03)
vec = normalized_feature_vector(times; t_start=0.0, t_end=0.1, max_density=200.0)
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
