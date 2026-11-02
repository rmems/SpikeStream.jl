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

Event types:
- `SpikeEvent` — one spike: time + neuron/unit id
- `SpikeTrain` — sorted spike times of one unit
- `spike_events` / `spike_trains` — convert between trains and events
- `spike_times` / `neuron_id` — accessors; `spike_times(train)` feeds the vector APIs above

"""
module SpikeStream

using Statistics

include("spike_features.jl")
include("spike_events.jl")

export spike_count, spike_density, isi_stats, detect_bursts
export windowed_spike_features, normalized_feature_vector
export SpikeFeatureSnapshot, spike_features, SpikeFeatureNormalizer, transform
export SpikeEvent, SpikeTrain, spike_times, neuron_id, spike_events, spike_trains

end # module
