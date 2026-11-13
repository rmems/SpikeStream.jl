# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    SpikeFeatureStream{I<:Integer}
    SpikeFeatureStream(; window, step=window, t_start=nothing, max_isi=0.02, min_spikes=3)

Stateful, bounded-memory feature extractor that incrementally ingests
[`SpikeEvent`](@ref)s and reports canonical [`SpikeFeatureSnapshot`](@ref)s per
neuron over stepped time windows.

# Windows

The window grid is anchored at `t_start` when given, otherwise at the first
accepted event. Window `k` covers `[t0 + k*step, t0 + k*step + window)` — the
same half-open interior semantics as [`windowed_spike_features`](@ref), so
`snapshot(stream)` matches the equivalent batch window. `step == window` gives
tumbling (non-overlapping) windows; `step < window` gives sliding windows.

`snapshot(stream)` reports the active window containing the newest accepted
event. The window may still be partial (more events can arrive), and when
`step > window` an event falling in a gap between windows reports the window
indexed by its step cell — consistent with batch behavior, where gap events
belong to no window.

# Ordering and memory

Events must arrive in non-decreasing time order. A `SpikeEvent` with
`t < ` the newest accepted time is late/out-of-order: it is dropped,
[`push!`](@ref) returns `false`, and the drop is counted in `stream.dropped`.
Times exactly equal to the last accepted time are accepted (simultaneous spikes
across units are normal).

State is bounded by the configured window, not the input history: per-neuron
time buffers are evicted once they fall behind the active window's start, and
neurons with no remaining in-horizon events are removed. Memory therefore
grows with the number of *active* units and the spike rate inside the window
horizon, not with stream length.

# Examples

```julia
julia> stream = SpikeFeatureStream(window = 0.1, step = 0.01);

julia> push!(stream, SpikeEvent(0.101, 17));
true

julia> push!(stream, SpikeEvent(0.104, 83));
true

julia> push!(stream, SpikeEvent(0.109, 17));
true

julia> push!(stream, SpikeEvent(0.105, 17));  # late: dropped
false

julia> snapshot(stream)
(t_start = 0.101, t_end = 0.201, window = 0, neurons = [...])
```

The `I` parameter fixes how `neuron_id`s are stored (default `Int`). Push events
whose ids fit `I`, or construct `SpikeFeatureStream{I}(; ...)` for other
integer id types.
"""
mutable struct SpikeFeatureStream{I<:Integer}
    window::Float64
    step::Float64
    max_isi::Float64
    min_spikes::Int
    t0::Float64
    anchored::Bool
    last_time::Float64
    times::Dict{I,Vector{Float64}}
    dropped::Int
    evict_k::Int
end

function SpikeFeatureStream{I}(;
    window::Real,
    step::Real = window,
    t_start = nothing,
    max_isi::Real = 0.02,
    min_spikes::Integer = 3,
) where {I<:Integer}
    window > 0 || throw(ArgumentError("window must be > 0"))
    step > 0 || throw(ArgumentError("step must be > 0"))
    if !isnothing(t_start)
        isfinite(t_start) || throw(ArgumentError("t_start must be finite"))
    end
    return SpikeFeatureStream{I}(
        Float64(window),
        Float64(step),
        Float64(max_isi),
        Int(min_spikes),
        isnothing(t_start) ? 0.0 : Float64(t_start),
        !isnothing(t_start),
        -Inf,
        Dict{I,Vector{Float64}}(),
        0,
        -1,
    )
end

"""
    SpikeFeatureStream(; window, step=window, ...)

Convenience constructor with `I = Int` for `neuron_id` storage; see
[`SpikeFeatureStream{I}`](@ref) for the full parameter list.
"""
SpikeFeatureStream(; kwargs...) = SpikeFeatureStream{Int}(; kwargs...)

function _push_event!(stream::SpikeFeatureStream{I}, t::Float64, id::I) where {I<:Integer}
    if stream.anchored
        t < stream.last_time && (stream.dropped += 1; return false)
    else
        stream.t0 = t
        stream.anchored = true
    end
    stream.last_time = t
    push!(get!(stream.times, id, Float64[]), t)
    return true
end

function _stream_window_end(t0::Float64, k::Int, step::Float64, window::Float64)
    ratio = window / step
    nearest = round(ratio)
    if abs(ratio - nearest) <= 32 * eps(Float64) * max(abs(ratio), 1.0)
        return t0 + (k + Int(nearest)) * step
    end
    return t0 + k * step + window
end

function _evict!(stream::SpikeFeatureStream{I}) where {I<:Integer}
    stream.last_time == -Inf && return nothing
    k = max(0, _start_index(stream.last_time, stream.t0, stream.step))
    k == stream.evict_k && return nothing
    stream.evict_k = k
    cur = stream.t0 + k * stream.step
    nxt = _stream_window_end(stream.t0, k, stream.step, stream.window)
    threshold = cur - _fp_ulp_band(cur, nxt, stream.t0, stream.step, stream.window)
    for id in collect(keys(stream.times))
        v = stream.times[id]
        i = searchsortedfirst(v, threshold)
        i > 1 && deleteat!(v, 1:(i-1))
        isempty(v) && delete!(stream.times, id)
    end
    return nothing
end

"""
    push!(stream::SpikeFeatureStream, event::SpikeEvent) -> Bool

Ingest one spike event. Returns `true` when the event was accepted and `false`
when it was dropped as late/out-of-order (`t` earlier than the newest accepted
time); dropped events are counted in `stream.dropped`.
"""
function Base.push!(stream::SpikeFeatureStream{I}, ev::SpikeEvent) where {I<:Integer}
    accepted = _push_event!(stream, Float64(ev.time), convert(I, ev.neuron_id))
    accepted && _evict!(stream)
    return accepted
end

"""
    push!(stream::SpikeFeatureStream, train::SpikeTrain) -> Int

Ingest all events of a [`SpikeTrain`](@ref) in one call, returning the number
of events accepted. Events earlier than the stream's newest accepted time are
dropped under the same late-event policy as [`push!`](@ref) with a
[`SpikeEvent`](@ref).
"""
function Base.push!(stream::SpikeFeatureStream{I}, train::SpikeTrain) where {I<:Integer}
    id = convert(I, train.neuron_id)
    accepted = 0
    for t in train.times
        _push_event!(stream, Float64(t), id) && (accepted += 1)
    end
    _evict!(stream)
    return accepted
end

function _stream_neuron_snapshot(
    id,
    v::AbstractVector{Float64},
    k::Int,
    stream::SpikeFeatureStream,
    cur::Float64,
    nxt::Float64,
    span_steps::Union{Int,Nothing},
)
    wtimes = Float64[
        t for t in v if _spike_in_window(
            t,
            k,
            stream.t0,
            stream.step,
            typemax(Int),
            Inf,
            cur,
            nxt,
            stream.window,
            false,
            span_steps,
        )
    ]
    count = length(wtimes)
    count == 0 && return nothing
    density = nxt > cur ? count / (nxt - cur) : 0.0
    stats = isi_stats_sorted(wtimes)
    bursts = length(
        detect_bursts_sorted(
            wtimes;
            max_isi = stream.max_isi,
            min_spikes = stream.min_spikes,
        ),
    )
    return (
        neuron_id = id,
        features = SpikeFeatureSnapshot(count, density, stats.mean, stats.cv, bursts),
    )
end

"""
    snapshot(stream::SpikeFeatureStream) -> NamedTuple

Return the feature snapshot of the active window — the step-grid window
containing the newest accepted event. Returns `nothing` before the first event
is accepted.

The result is a named tuple:

- `t_start`, `t_end` — the window bounds (`[t_start, t_end)` semantics, same as
  [`windowed_spike_features`](@ref))
- `window` — the window index `k` on the stream's step grid
- `neurons` — a `Vector` sorted by `neuron_id`, each element a named tuple
  `(neuron_id, features)` where `features` is a [`SpikeFeatureSnapshot`](@ref)
  computed with the stream's `max_isi`/`min_spikes`. Only neurons with at least
  one spike inside the window appear.

Only buffered in-window events are scanned, so snapshot cost is bounded by the
window horizon rather than stream history.
"""
function snapshot(stream::SpikeFeatureStream)
    stream.last_time == -Inf && return nothing
    k = max(0, _start_index(stream.last_time, stream.t0, stream.step))
    cur = stream.t0 + k * stream.step
    nxt = _stream_window_end(stream.t0, k, stream.step, stream.window)
    span_steps = _window_span_steps(stream.window, stream.step)

    rows = NamedTuple[]
    for (id, v) in stream.times
        row = _stream_neuron_snapshot(id, v, k, stream, cur, nxt, span_steps)
        row === nothing || push!(rows, row)
    end
    sort!(rows; by = row -> row.neuron_id)

    return (t_start = cur, t_end = nxt, window = k, neurons = rows)
end
