# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    SpikeStream

Spike-stream feature extraction primitives for spiking neural systems.

Primary package boundary:
- `spike_count` — spike count in full stream or a time window
- `spike_density` — spikes per unit time
- `isi_stats` — inter-spike interval summary statistics
- `detect_bursts` — burst detection from short ISIs
- `windowed_spike_features` — rolling/windowed extraction
- `spike_features` / `SpikeFeatureSnapshot` — canonical raw feature snapshot
- `SpikeFeatureNormalizer` / `transform` — explicit feature normalization
- `normalized_feature_vector` — compatibility helper (snapshot + default normalizer)
- `SpikeFeatureStream` / `snapshot` — bounded-memory incremental `SpikeEvent` ingest
- `population_spike_features` / `PopulationFeatureSnapshot` — population-level
  activity summary over `SpikeEvent`/`SpikeTrain` collections or a stream snapshot
- `unit_firing_rates` — per-unit firing rates for a population

Event types:
- `SpikeEvent` — one spike: time + neuron/unit id
- `SpikeTrain` — sorted spike times of one unit
- `spike_events` / `spike_trains` — convert between trains and events
- `spike_times` / `neuron_id` — accessors; `spike_times(train)` feeds the vector APIs above

Synchrony metrics (`tolerance` = max `|Δt|` between coincident spikes):
- `coincidence_count` — matched coincidence pairs between two trains/vectors
- `coincidence_rate` — fraction of spikes in a coincidence pair, in `[0, 1]`
- `pairwise_synchrony` — per-unit-pair coincidence summary, sorted by unit id
- `population_synchrony` — mean pairwise coincidence rate (`NaN` for `< 2` units)

"""
module SpikeStream

using Statistics

include("spike_features.jl")
include("spike_events.jl")
include("spike_synchrony.jl")
include("spike_stream.jl")
include("population_features.jl")

export spike_count, spike_density, isi_stats, detect_bursts
export windowed_spike_features, normalized_feature_vector
export SpikeFeatureSnapshot, spike_features, SpikeFeatureNormalizer, transform
export SpikeEvent, SpikeTrain, spike_times, neuron_id, spike_events, spike_trains
export SpikeFeatureStream, snapshot
export PopulationFeatureSnapshot, population_spike_features, unit_firing_rates
export coincidence_count, coincidence_rate, pairwise_synchrony, population_synchrony

end # module
