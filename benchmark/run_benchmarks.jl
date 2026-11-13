# SPDX-License-Identifier: MIT OR Apache-2.0

using BenchmarkTools
using SpikeStream
using Random

Random.seed!(42)

function generate_spike_train(n::Int, duration::Float64)
    return sort!(rand(n) .* duration)
end

const SCALES = [("small", 100, 1.0), ("medium", 1_000, 10.0), ("large", 10_000, 100.0)]

const SUITE = BenchmarkGroup()

for (label, n, dur) in SCALES
    spikes = generate_spike_train(n, dur)
    group = BenchmarkGroup()

    group["spike_count"] = @benchmarkable spike_count($spikes)
    group["spike_count_window"] =
        @benchmarkable spike_count($spikes; t_start = 0.0, t_end = $(dur / 2))
    group["spike_density"] = @benchmarkable spike_density($spikes)
    group["spike_density_window"] =
        @benchmarkable spike_density($spikes; t_start = 0.0, t_end = $(dur / 2))
    group["isi_stats"] = @benchmarkable isi_stats($spikes)
    group["detect_bursts"] = @benchmarkable detect_bursts($spikes)
    group["windowed_spike_features"] = @benchmarkable windowed_spike_features(
        $spikes;
        window_size = $(dur / 100),
        step = $(dur / 100),
    )
    group["normalized_feature_vector"] = @benchmarkable normalized_feature_vector(
        $spikes;
        t_start = 0.0,
        t_end = $dur,
        max_density = 1000.0,
    )

    SUITE[label] = group
end

# Streaming (GH#30): incremental ingest vs repeated batch recomputation.
# The stream keeps per-neuron state bounded by the window horizon; the batch
# baselines rescan the full event history on every call.
stream_events = sort!(SpikeEvent[SpikeEvent(rand() * 10.0, rand(1:16)) for _ = 1:5_000])
stream_times = Float64[ev.time for ev in stream_events]
recompute_at = round.(Int, range(500, 5_000, length = 10))

stream_group = BenchmarkGroup()
stream_group["stream_ingest+snapshot"] = @benchmarkable begin
    s = SpikeFeatureStream(window = 0.5, step = 0.1)
    foreach(ev -> push!(s, ev), $stream_events)
    snapshot(s)
end
stream_group["batch_windowed_features"] =
    @benchmarkable windowed_spike_features($stream_times; window_size = 0.5, step = 0.1)
stream_group["batch_recompute_prefixes"] = @benchmarkable begin
    for m in $recompute_at
        spike_features($stream_times[1:m]; t_start = 0.0, t_end = 10.0)
    end
end
SUITE["stream"] = stream_group

results = run(SUITE, verbose = true, seconds = 5)

println()
println("=" ^ 70)
println("  SpikeStream.jl Benchmark Results")
println("=" ^ 70)

for (label, n, _) in SCALES
    group = results[label]
    println()
    println("  $label ($n spikes)")
    println("  " * "-" ^ 60)
    for name in sort(collect(keys(group)))
        trial = group[name]
        m = median(trial)
        time_str = BenchmarkTools.prettytime(time(m))
        alloc_str = allocs(m)
        mem_str = BenchmarkTools.prettymemory(memory(m))
        println("  $(rpad(name, 32)) $time_str  ($alloc_str allocs, $mem_str)")
    end
end

group = results["stream"]
println()
println("  stream (5_000 events, 16 units; incremental vs batch)")
println("  " * "-" ^ 60)
for name in sort(collect(keys(group)))
    trial = group[name]
    m = median(trial)
    time_str = BenchmarkTools.prettytime(time(m))
    alloc_str = allocs(m)
    mem_str = BenchmarkTools.prettymemory(memory(m))
    println("  $(rpad(name, 32)) $time_str  ($alloc_str allocs, $mem_str)")
end

println()
println("=" ^ 70)
