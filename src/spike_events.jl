# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    SpikeEvent{T<:Real,I<:Integer}(time, neuron_id)
    SpikeEvent(time::Real, neuron_id::Integer)

A single spike: an event `time` emitted by the unit `neuron_id`.

`time` must be finite (`NaN`/`±Inf` throw `ArgumentError`) and may use any `Real`
type (`Float64`, `Float32`, `Int`, `Rational`, ...) in any consistent unit.
`neuron_id` is an opaque integer label; SpikeStream assigns it no meaning beyond
identity and ordering, so negative or zero ids are allowed.

Events order by `time`, then by `neuron_id` (`isless`), so `sort(events)` yields a
deterministic time-major order even when several units spike at the same instant.

# Examples
```julia
julia> ev = SpikeEvent(0.25, 3)
SpikeEvent{Float64, Int64}(0.25, 3)

julia> isless(SpikeEvent(0.1, 7), SpikeEvent(0.1, 9))
true
```
"""
struct SpikeEvent{T<:Real,I<:Integer}
    time::T
    neuron_id::I

    function SpikeEvent{T,I}(time::Real, neuron_id::Integer) where {T<:Real,I<:Integer}
        # Check the input first so NaN/±Inf raise ArgumentError even when T
        # cannot represent them; narrowing can also turn a finite input into
        # ±Inf, so the converted value is checked too.
        isfinite(time) || throw(ArgumentError("spike times must be finite"))
        t = convert(T, time)
        isfinite(t) || throw(ArgumentError("spike times must be finite"))
        return new{T,I}(t, neuron_id)
    end
end

SpikeEvent(time::T, neuron_id::I) where {T<:Real,I<:Integer} =
    SpikeEvent{T,I}(time, neuron_id)

Base.convert(::Type{SpikeEvent{T,I}}, ev::SpikeEvent) where {T<:Real,I<:Integer} =
    SpikeEvent{T,I}(ev.time, ev.neuron_id)
Base.convert(::Type{SpikeEvent{T,I}}, ev::SpikeEvent{T,I}) where {T<:Real,I<:Integer} = ev

Base.promote_rule(
    ::Type{SpikeEvent{T1,I1}},
    ::Type{SpikeEvent{T2,I2}},
) where {T1<:Real,I1<:Integer,T2<:Real,I2<:Integer} =
    SpikeEvent{promote_type(T1, T2),promote_type(I1, I2)}

Base.isless(a::SpikeEvent, b::SpikeEvent) =
    isless((a.time, a.neuron_id), (b.time, b.neuron_id))

"""
    SpikeTrain(neuron_id::Integer, times::AbstractVector{<:Real}; sorted::Bool=false)
    SpikeTrain{T,I}(neuron_id, times)

Spike times of a single unit `neuron_id`, stored in non-decreasing order.

Invariants (checked by every constructor; violations throw `ArgumentError`):

- every time is finite;
- times are sorted non-decreasing. Duplicate timestamps are allowed and kept,
  matching the zero-length ISIs accepted by the timestamp-vector APIs.

Construction and allocation:

- `SpikeTrain(neuron_id, times)` copies `times` into a new `Vector{eltype(times)}` and
  sorts the copy, so unsorted input is accepted and the caller's vector is never
  mutated or aliased. One allocation for the copy (plus sort scratch, if any).
- `SpikeTrain(neuron_id, times; sorted=true)` still copies but skips sorting; it only
  checks the order in one pass and throws if `times` is not sorted.
- `SpikeTrain{T,I}(neuron_id, times)` is the no-copy path: when `times isa Vector{T}`
  the train aliases it (zero allocations); otherwise it is `convert`ed to `Vector{T}`.
  Unsorted input throws. Mutating an aliased vector afterwards can break the
  invariants, so only use it for vectors you hand over.

The timestamp-vector APIs accept a train's times directly, e.g.
`spike_count(spike_times(train))`.

# Examples
```julia
julia> train = SpikeTrain(2, [0.3, 0.1, 0.2]);

julia> spike_times(train)
3-element Vector{Float64}:
 0.1
 0.2
 0.3

julia> neuron_id(train)
2
```
"""
struct SpikeTrain{T<:Real,I<:Integer}
    neuron_id::I
    times::Vector{T}

    function SpikeTrain{T,I}(
        neuron_id::Integer,
        times::AbstractVector{<:Real},
    ) where {T<:Real,I<:Integer}
        for t in times
            isfinite(t) || throw(ArgumentError("spike times must be finite"))
        end
        v = convert(Vector{T}, times)
        _check_train_times(v)
        return new{T,I}(neuron_id, v)
    end
end

function SpikeTrain(
    neuron_id::I,
    times::AbstractVector{T};
    sorted::Bool = false,
) where {T<:Real,I<:Integer}
    v = Vector{T}(undef, length(times))
    copyto!(v, times)
    sorted || sort!(v)
    return SpikeTrain{T,I}(neuron_id, v)
end

function _check_train_times(times::AbstractVector{<:Real})
    for i in eachindex(times)
        t = times[i]
        isfinite(t) || throw(ArgumentError("spike times must be finite"))
        if i > firstindex(times) && isless(t, times[prevind(times, i)])
            throw(ArgumentError("SpikeTrain times must be sorted in non-decreasing order"))
        end
    end
    return nothing
end

Base.:(==)(a::SpikeTrain, b::SpikeTrain) = a.neuron_id == b.neuron_id && a.times == b.times
Base.hash(train::SpikeTrain, h::UInt) =
    hash(train.times, hash(train.neuron_id, hash(:SpikeTrain, h)))

Base.length(train::SpikeTrain) = length(train.times)
Base.isempty(train::SpikeTrain) = isempty(train.times)

"""
    spike_times(train::SpikeTrain) -> Vector

Return the sorted spike times of `train`. This is the train's own storage (no copy,
zero allocations); do not mutate it.
"""
spike_times(train::SpikeTrain) = train.times

"""
    neuron_id(x::Union{SpikeEvent,SpikeTrain}) -> Integer

Return the unit identity of a spike event or spike train.
"""
neuron_id(ev::SpikeEvent) = ev.neuron_id
neuron_id(train::SpikeTrain) = train.neuron_id

"""
    spike_events(train::SpikeTrain) -> Vector{SpikeEvent}
    spike_events(trains::AbstractVector{<:SpikeTrain}) -> Vector{SpikeEvent}

Expand trains into events. For one train the result is already sorted and costs one
allocation of `length(train)` events. For several trains the events are concatenated
into one preallocated vector and sorted with `isless` (time, then `neuron_id`); the
sort may allocate scratch space. Trains with different time or id types are promoted
to a common `SpikeEvent` type (an empty, abstractly typed input gives
`SpikeEvent{Float64,Int}[]`).
"""
function spike_events(train::SpikeTrain{T,I}) where {T,I}
    times = train.times
    events = Vector{SpikeEvent{T,I}}(undef, length(times))
    for i in eachindex(times, events)
        events[i] = SpikeEvent{T,I}(times[i], train.neuron_id)
    end
    return events
end

function spike_events(trains::AbstractVector{<:SpikeTrain})
    S = eltype(trains)
    if isconcretetype(S)
        E = SpikeEvent{eltype(fieldtype(S, :times)),fieldtype(S, :neuron_id)}
    elseif isempty(trains)
        E = SpikeEvent{Float64,Int}
    else
        T = mapreduce(tr -> eltype(tr.times), promote_type, trains)
        I = mapreduce(tr -> typeof(tr.neuron_id), promote_type, trains)
        E = SpikeEvent{T,I}
    end
    return _merge_events(E, trains)
end

function _merge_events(
    ::Type{SpikeEvent{T,I}},
    trains::AbstractVector{<:SpikeTrain},
) where {T,I}
    events = Vector{SpikeEvent{T,I}}(undef, sum(length, trains; init = 0))
    k = 0
    for train in trains
        id = convert(I, train.neuron_id)
        for t in train.times
            k += 1
            events[k] = SpikeEvent{T,I}(t, id)
        end
    end
    return sort!(events)
end

"""
    spike_trains(events::AbstractVector{<:SpikeEvent}) -> Vector{SpikeTrain}

Group events by `neuron_id` into one train per unit, ordered by ascending
`neuron_id`. Events may arrive in any order; each train's times are sorted and
duplicates are kept. Units with no events do not appear in the result. Events with
different time or id types are promoted to a common type first (an empty, abstractly
typed input gives `SpikeTrain{Float64,Int}[]`). Allocates a sort permutation plus one
times vector per unit.
"""
function spike_trains(events::AbstractVector{<:SpikeEvent})
    E = eltype(events)
    if !isconcretetype(E)
        E =
            isempty(events) ? SpikeEvent{Float64,Int} :
            mapreduce(typeof, promote_type, events)
    end
    return _group_trains(E, events)
end

function _group_trains(
    ::Type{SpikeEvent{T,I}},
    events::AbstractVector{<:SpikeEvent},
) where {T,I}
    order = sortperm(events; by = ev -> (ev.neuron_id, ev.time))
    trains = SpikeTrain{T,I}[]
    n = length(order)
    i = 1
    while i <= n
        id = events[order[i]].neuron_id
        j = i
        while j < n && events[order[j+1]].neuron_id == id
            j += 1
        end
        times = T[events[order[k]].time for k = i:j]
        push!(trains, SpikeTrain{T,I}(id, times))
        i = j + 1
    end
    return trains
end
