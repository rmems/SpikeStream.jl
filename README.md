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
- `population_spike_features(population; t_start, t_end, n_total_units)` → [`PopulationFeatureSnapshot`](#population-features)
- `unit_firing_rates(population; t_start, t_end)` → per-unit rates over the window
- `coincidence_count(a, b; tolerance)` / `coincidence_rate(a, b; tolerance)`
- `pairwise_synchrony(population; tolerance)` / `population_synchrony(population; tolerance)`

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

## Synchrony metrics

Lightweight coincidence/synchrony metrics over `SpikeTrain` collections or
`SpikeEvent` lists (#32). Two spikes **coincide** when `|t_a - t_b| <= tolerance`
(inclusive; `tolerance` defaults to `0.005` in the input's time units and must be
`>= 0`). Matching is one-to-one: each spike belongs to at most one coincidence
pair, and the count is the maximum such matching — computed with a linear
two-pointer sweep (`O(n_a + n_b)` per pair), not an all-pairs scan.

- `coincidence_count(a, b; tolerance)` → number of matched coincidence pairs (`>= 0`)
- `coincidence_rate(a, b; tolerance)` → `2*count/(n_a + n_b)` in `[0, 1]` — the fraction of spikes taking part in a coincidence; `0` when either side is empty
- `pairwise_synchrony(population; tolerance, t_start, t_end)` → one `(id_a, id_b, coincidence_count, coincidence_rate)` row per unordered unit pair, sorted by unit id. `population` is a vector of `SpikeTrain`s (same-id trains merge) or `SpikeEvent`s (grouped by id); sparse/non-contiguous ids are fine, silent units report `0`s
- `population_synchrony(population; tolerance, t_start, t_end)` → mean pairwise `coincidence_rate` in `[0, 1]`; `NaN` when fewer than 2 units are observed

Optional `t_start`/`t_end` restrict the comparison to spikes inside the inclusive
window `[t_start, t_end]` (same bounds as `spike_count`). That is also the
streaming path until a dedicated synchrony stream exists: group events per
window and call the batch functions on each slice.

```julia
trains = [SpikeTrain(1, [0.10, 0.20]), SpikeTrain(7, [0.11, 0.60]), SpikeTrain(-3, Float64[])]
rows = pairwise_synchrony(trains; tolerance = 0.02)
# pairs: (-3,1) -> 0, (-3,7) -> 0, (1,7) -> count 1, rate 0.5
population_synchrony(trains; tolerance = 0.02)  # ≈ 0.167, mean over the 3 pairs
```

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
- `PopulationFeatureSnapshot` — unit counts and `total_spike_count` are integers `>= 0`; `population_rate`, `mean_unit_rate`, `count_mean`, `count_variance` are `>= 0`; `active_fraction` is in `[0, 1]`; `fano_factor` is `>= 0` or `NaN` when `count_mean == 0`
- `unit_firing_rates` → vector of `(neuron_id, spike_count, firing_rate)` rows sorted by id; `firing_rate >= 0`
- `coincidence_count` → integer `>= 0`
- `coincidence_rate` → real in `[0, 1]` (`0` when either input is empty)
- `pairwise_synchrony` → vector of `(id_a, id_b, coincidence_count >= 0, coincidence_rate in [0, 1])` rows sorted by unit id
- `population_synchrony` → real in `[0, 1]`, or `NaN` for `< 2` observed units

## Package Boundary

SpikeStream.jl owns spike-stream **feature extraction** only. Kinetic / signal-dynamics APIs live in the Rust sibling [rmems/kinetic-signals](https://github.com/rmems/kinetic-signals).

| Area | Owner | Notes |
| --- | --- | --- |
| `spike_count`, `spike_density`, `isi_stats`, `detect_bursts`, `windowed_spike_features`, `spike_features`, `SpikeFeatureNormalizer` / `transform`, `normalized_feature_vector`, `SpikeFeatureStream` / `snapshot`, `population_spike_features` / `PopulationFeatureSnapshot`, `unit_firing_rates`, `coincidence_count`, `coincidence_rate`, `pairwise_synchrony`, `population_synchrony` | **SpikeStream.jl** | Public API of this package |
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

## Population features

`population_spike_features` summarizes activity across a whole unit population
into a `PopulationFeatureSnapshot`. The `population` argument may be a vector
of `SpikeTrain`s, a flat vector of `SpikeEvent`s, a single `SpikeTrain`, or a
`SpikeFeatureStream` (which aggregates its current `snapshot` and returns
`nothing` before the first accepted event).

```julia
pop = [SpikeTrain(1, [0.1, 0.2, 0.3]), SpikeTrain(7, [0.15]), SpikeTrain(1003, Float64[])]
snap = population_spike_features(pop; t_start=0.0, t_end=1.0)
snap.population_rate    # total spikes / window duration -> 4.0
snap.active_fraction   # n_active_units / n_total_units -> 2/3 (sparsity = 1 - this)
snap.count_mean        # mean per-unit spike count -> 4/3
snap.count_variance    # uncorrected variance over n_total_units slots
snap.fano_factor       # count_variance / count_mean (NaN when count_mean == 0)

rates = unit_firing_rates(pop; t_start=0.0, t_end=1.0)
# (neuron_id, spike_count, firing_rate) rows sorted by id
```

Denominator semantics are explicit: `n_observed_units` counts distinct ids in
the input (an empty `SpikeTrain` is observed but silent; a `SpikeEvent` list
only observes ids that appear), `n_active_units` counts observed units with at
least one in-window spike, and `n_total_units` is the denominator used for
`mean_unit_rate`, `active_fraction`, `count_mean` and `count_variance` — it
defaults to `n_observed_units` and may be passed larger to account for
declared-but-absent units. Neuron ids are opaque: sparse, negative, or
non-contiguous ids need no dense range. Non-positive window durations report
counts with all rate fields `0`; `fano_factor` is `NaN` when `count_mean == 0`
(no meaningful Fano factor for a spikeless window or empty population).

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
