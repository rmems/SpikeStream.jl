# SPDX-License-Identifier: MIT OR Apache-2.0

const SynchronyPairRow = @NamedTuple{
    id_a::Integer,
    id_b::Integer,
    coincidence_count::Int,
    coincidence_rate::Float64,
}

const _SyncPair = Union{SpikeTrain,AbstractVector{<:Real}}
const _SyncPopulation = Union{AbstractVector{<:SpikeTrain},AbstractVector{<:SpikeEvent}}

function _check_sync_tolerance(tolerance)::Float64
    tol = Float64(tolerance)
    (isfinite(tol) && tol >= 0) ||
        throw(ArgumentError("tolerance must be a finite value >= 0"))
    return tol
end

function _check_sync_bound(value, name)::Float64
    v = Float64(value)
    isfinite(v) || throw(ArgumentError("$name must be finite"))
    return v
end

_pair_times(times::AbstractVector{<:Real}) = _sorted_finite_times(times)
_pair_times(train::SpikeTrain{Float64}) = spike_times(train)
function _pair_times(train::SpikeTrain)
    times = Float64.(spike_times(train))
    all(isfinite, times) || throw(ArgumentError("spike times must be finite"))
    return times
end

function _coincidence_pairs(
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    tol::Float64,
)::Int
    i, j = firstindex(a), firstindex(b)
    count = 0
    while i <= lastindex(a) && j <= lastindex(b)
        d = a[i] - b[j]
        if d > tol
            j += 1
        elseif d < -tol
            i += 1
        else
            count += 1
            i += 1
            j += 1
        end
    end
    return count
end

_rate_from(count::Int, n_a::Int, n_b::Int)::Float64 =
    n_a + n_b > 0 ? 2 * count / (n_a + n_b) : 0.0

function _unit_time_map(
    trains::AbstractVector{<:SpikeTrain};
    t_start = nothing,
    t_end = nothing,
)
    lo = isnothing(t_start) ? -Inf : _check_sync_bound(t_start, "t_start")
    hi = isnothing(t_end) ? Inf : _check_sync_bound(t_end, "t_end")
    grouped = Dict{Integer,Vector{Float64}}()
    for tr in trains
        v = get!(grouped, tr.neuron_id, Float64[])
        left = searchsortedfirst(tr.times, lo)
        right = searchsortedlast(tr.times, hi)
        for k = left:right
            t = Float64(tr.times[k])
            isfinite(t) || throw(ArgumentError("spike times must be finite"))
            push!(v, t)
        end
    end
    ids = sort!(collect(keys(grouped)))
    times = Vector{Float64}[sort!(grouped[id]) for id in ids]
    return ids, times
end

function _unit_time_map(
    events::AbstractVector{<:SpikeEvent};
    t_start = nothing,
    t_end = nothing,
)
    lo = isnothing(t_start) ? -Inf : _check_sync_bound(t_start, "t_start")
    hi = isnothing(t_end) ? Inf : _check_sync_bound(t_end, "t_end")
    grouped = Dict{Integer,Vector{Float64}}()
    for ev in events
        v = get!(grouped, ev.neuron_id, Float64[])
        t = Float64(ev.time)
        isfinite(t) || throw(ArgumentError("spike times must be finite"))
        lo <= t <= hi && push!(v, t)
    end
    ids = sort!(collect(keys(grouped)))
    times = Vector{Float64}[sort!(grouped[id]) for id in ids]
    return ids, times
end

"""
    coincidence_count(a, b; tolerance=0.005) -> Int

Count coincident spike pairs between two trains. `a` and `b` may each be a
[`SpikeTrain`](@ref) or an `AbstractVector{<:Real}` of spike times (unsorted
vectors are sorted and checked like the other timestamp-vector APIs).

Two spikes **coincide** when `|t_a - t_b| <= tolerance`. `tolerance` must be a
finite value `>= 0` in the same time unit as the inputs (default `0.005`,
i.e. 5 ms when times are seconds); `tolerance = 0` counts only exactly
simultaneous spikes.

Matching is one-to-one: every spike belongs to at most one coincidence pair,
and the count is the maximum number of disjoint pairs. It is computed
greedily in a single two-pointer sweep over the sorted times, `O(n_a + n_b)`
per call — not an all-pairs nested scan.

```julia
julia> coincidence_count(SpikeTrain(1, [0.10, 0.20]), SpikeTrain(2, [0.105]))
1
```
"""
function coincidence_count(a::_SyncPair, b::_SyncPair; tolerance::Real = 0.005)
    tol = _check_sync_tolerance(tolerance)
    return _coincidence_pairs(_pair_times(a), _pair_times(b), tol)
end

"""
    coincidence_rate(a, b; tolerance=0.005) -> Float64

Normalized coincidence score: the fraction of spikes that participate in a
coincidence pair, `2 * coincidence_count / (n_a + n_b)` in `[0, 1]`.

`1.0` means every spike in both trains is part of a coincidence pair (true in
particular for identical trains). Returns `0.0` when either input has no
spikes — silence is `0`, not an error. Same `tolerance` semantics as
[`coincidence_count`](@ref).
"""
function coincidence_rate(a::_SyncPair, b::_SyncPair; tolerance::Real = 0.005)
    tol = _check_sync_tolerance(tolerance)
    ta = _pair_times(a)
    tb = _pair_times(b)
    return _rate_from(_coincidence_pairs(ta, tb, tol), length(ta), length(tb))
end

"""
    pairwise_synchrony(population; tolerance=0.005, t_start=nothing, t_end=nothing) -> Vector

Per-unit-pair coincidence summary for a population of units. `population` may
be an `AbstractVector{<:SpikeTrain}` (trains sharing a `neuron_id` are merged
into one unit; an empty train is still an observed, silent unit) or an
`AbstractVector{<:SpikeEvent}` in any order (grouped by `neuron_id`; only the
ids that appear are observed). A single `SpikeTrain` is a one-unit population
and yields an empty result.

Returns one `(id_a, id_b, coincidence_count, coincidence_rate)` named tuple
per unordered unit pair, sorted by `(id_a, id_b)`. Unit ids are opaque labels:
sparse, non-contiguous, zero or negative ids are reported as-is, and no dense
id range is assumed. `coincidence_count` and `coincidence_rate` follow
[`coincidence_count`](@ref) semantics with the given `tolerance`; pairs
involving silent units report zeros.

`t_start`/`t_end` restrict the comparison to spikes inside the inclusive
window `[t_start, t_end]` (the same bounds [`spike_count`](@ref) uses). Units
with no in-window spikes still appear, as observed-silent pairs. Because
in-window times are sliced by binary search and each pair is matched in a
single linear sweep, cost is bounded by in-window spike counts
(`O(Σ (n_a + n_b))` over all pairs), never `O(N^2 * history)`.
"""
function pairwise_synchrony(
    population::_SyncPopulation;
    tolerance::Real = 0.005,
    t_start = nothing,
    t_end = nothing,
)
    tol = _check_sync_tolerance(tolerance)
    ids, times = _unit_time_map(population; t_start = t_start, t_end = t_end)
    n = length(ids)
    rows = SynchronyPairRow[]
    sizehint!(rows, n * (n - 1) ÷ 2)
    for i = 1:(n-1)
        a = times[i]
        for j = (i+1):n
            c = _coincidence_pairs(a, times[j], tol)
            push!(
                rows,
                (
                    id_a = ids[i],
                    id_b = ids[j],
                    coincidence_count = c,
                    coincidence_rate = _rate_from(c, length(a), length(times[j])),
                ),
            )
        end
    end
    return rows
end

pairwise_synchrony(train::SpikeTrain; kwargs...) =
    pairwise_synchrony(SpikeTrain[train]; kwargs...)

"""
    population_synchrony(population; tolerance=0.005, t_start=nothing, t_end=nothing) -> Float64

Population-level synchrony aggregate: the mean [`coincidence_rate`](@ref) over
all unordered unit pairs of `population` (same input forms as
[`pairwise_synchrony`](@ref)).

The result lies in `[0, 1]`; `1.0` means every unit pair is fully coincident.
Pairs involving silent units contribute `0.0`, so a quiet population lowers
the mean. Returns `NaN` when fewer than two units are observed — there is no
pair to average over; a fully silent population of two or more units is `0.0`.
"""
function population_synchrony(
    population::_SyncPopulation;
    tolerance::Real = 0.005,
    t_start = nothing,
    t_end = nothing,
)
    tol = _check_sync_tolerance(tolerance)
    ids, times = _unit_time_map(population; t_start = t_start, t_end = t_end)
    n = length(ids)
    n < 2 && return NaN
    acc = 0.0
    for i = 1:(n-1), j = (i+1):n
        c = _coincidence_pairs(times[i], times[j], tol)
        acc += _rate_from(c, length(times[i]), length(times[j]))
    end
    return acc / (n * (n - 1) ÷ 2)
end

population_synchrony(train::SpikeTrain; kwargs...) =
    population_synchrony(SpikeTrain[train]; kwargs...)
