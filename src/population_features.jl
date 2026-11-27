# SPDX-License-Identifier: MIT OR Apache-2.0

const UnitRateRow = @NamedTuple{neuron_id::Integer, spike_count::Int, firing_rate::Float64}

"""
    PopulationFeatureSnapshot

Population-level spike features for one observation window, aggregated across
the units of a spike-event population. Built by
[`population_spike_features`](@ref); all fields are raw physical/statistical
quantities (no normalization applied).

# Denominator semantics

Three distinct unit counts are reported:

- `n_observed_units` — distinct unit ids present in the input collection. A
  `SpikeTrain` is observed even when it carries no spikes (a declared-silent
  unit); a `SpikeEvent` list only observes the ids that actually appear in it.
- `n_active_units` — observed units with at least one spike inside the window.
- `n_total_units` — the denominator: the declared population size used for
  `mean_unit_rate`, `active_fraction`, `count_mean` and `count_variance`. It
  defaults to `n_observed_units`; pass a larger `n_total_units` to count
  declared-but-absent units as zeros.

# Fields

- `duration` — observation-window length in the input's time units (`>= 0`)
- `n_total_units` — denominator population size (`>= n_observed_units`)
- `n_observed_units` — distinct unit ids present in the input
- `n_active_units` — units with `>= 1` spike in the window
- `total_spike_count` — spikes across all observed units in the window (`>= 0`)
- `population_rate` — aggregate event rate `total_spike_count / duration`
  (spikes per unit time across the whole population; `0` when `duration <= 0`)
- `mean_unit_rate` — mean per-unit firing rate `population_rate / n_total_units`;
  silent and absent units contribute `0`
- `active_fraction` — `n_active_units / n_total_units` in `[0, 1]`; population
  sparsity is `1 - active_fraction`
- `count_mean` — mean per-unit spike count `total_spike_count / n_total_units`
- `count_variance` — uncorrected variance of the per-unit spike counts over all
  `n_total_units` slots (silent and absent units count as `0`)
- `fano_factor` — `count_variance / count_mean`; `NaN` when `count_mean == 0`,
  since a spikeless window or empty population has no meaningful Fano factor
"""
struct PopulationFeatureSnapshot
    duration::Float64
    n_total_units::Int
    n_observed_units::Int
    n_active_units::Int
    total_spike_count::Int
    population_rate::Float64
    mean_unit_rate::Float64
    active_fraction::Float64
    count_mean::Float64
    count_variance::Float64
    fano_factor::Float64
end

function _resolve_n_total_units(n_total_units, n_observed::Int)::Int
    n = isnothing(n_total_units) ? n_observed : Int(n_total_units)
    n < 0 && throw(ArgumentError("n_total_units must be >= 0"))
    n < n_observed && throw(
        ArgumentError(
            "n_total_units ($n) cannot be smaller than the observed unit count ($n_observed)",
        ),
    )
    return n
end

function _population_duration(tmin, tmax, n_spikes::Int, t_start, t_end)::Float64
    if isnothing(t_start) || isnothing(t_end)
        # Same sparse-input rule as `spike_density`: an inferred edge is
        # undefined for fewer than two spikes.
        n_spikes < 2 && return 0.0
        lo = isnothing(t_start) ? tmin : t_start
        hi = isnothing(t_end) ? tmax : t_end
    else
        lo = t_start
        hi = t_end
    end
    (isnothing(lo) || isnothing(hi)) && return 0.0
    # Subtract in a precision that keeps the endpoints distinct: converting
    # to Float64 first can collapse large integer, rational or high-precision
    # (BigFloat) bounds to one value while the window still contains spikes.
    d = if lo isa Integer && hi isa Integer
        Float64(BigInt(hi) - BigInt(lo))
    elseif lo isa Union{Integer,Rational} && hi isa Union{Integer,Rational}
        Float64(Rational{BigInt}(hi) - Rational{BigInt}(lo))
    else
        Float64(BigFloat(hi) - BigFloat(lo))
    end
    return max(0.0, d)
end

# Window membership is evaluated in each bound's own precision: an
# AbstractFloat bound compares the Float64 images of spike times — the same
# domain `spike_count` uses after conversion — while an exact (integer or
# rational) bound compares the stored values directly, preserving timestamp
# precision beyond Float64.
_population_below_lo(t, lo) =
    lo isa AbstractFloat ? isless(Float64(t), Float64(lo)) : isless(t, lo)
_population_above_hi(t, hi) =
    hi isa AbstractFloat ? isless(Float64(hi), Float64(t)) : isless(hi, t)

function _population_window_counts(
    trains::AbstractVector{<:SpikeTrain};
    t_start = nothing,
    t_end = nothing,
)
    T = eltype(trains)
    counts = Dict{isconcretetype(T) ? fieldtype(T, :neuron_id) : Integer,Int}()
    unbounded = isnothing(t_start) && isnothing(t_end)
    need_extrema = isnothing(t_start) || isnothing(t_end)
    lo = isnothing(t_start) ? -Inf : t_start
    hi = isnothing(t_end) ? Inf : t_end
    empty_window = !unbounded && isless(hi, lo)
    tmin = nothing
    tmax = nothing
    n_spikes = 0
    for train in trains
        times = train.times
        n_spikes += length(times)
        c = if unbounded
            length(times)
        elseif empty_window
            0
        else
            # Train times are stored sorted; searching them directly avoids
            # re-sorting a copy. An AbstractFloat bound searches the Float64
            # image of the times (still sorted; same domain `spike_count`
            # uses), an exact bound searches the stored values.
            left = if lo isa AbstractFloat
                searchsortedfirst(times, lo; by = Float64)
            else
                searchsortedfirst(times, lo)
            end
            right = if hi isa AbstractFloat
                searchsortedlast(times, hi; by = Float64)
            else
                searchsortedlast(times, hi)
            end
            max(0, right - left + 1)
        end
        counts[train.neuron_id] = get(counts, train.neuron_id, 0) + c
        (isempty(times) || !need_extrema) && continue
        # `isless` keeps the original timestamp where `min`/`max` would
        # promote mixed types: a signed/unsigned pair throws on promotion,
        # and Float32/Int promotion can collapse a positive span.
        tmin = isnothing(tmin) || isless(first(times), tmin) ? first(times) : tmin
        tmax = isnothing(tmax) || isless(tmax, last(times)) ? last(times) : tmax
    end
    return counts, tmin, tmax, n_spikes
end

function _population_window_counts(
    events::AbstractVector{<:SpikeEvent};
    t_start = nothing,
    t_end = nothing,
)
    E = eltype(events)
    counts = Dict{isconcretetype(E) ? fieldtype(E, :neuron_id) : Integer,Int}()
    need_extrema = isnothing(t_start) || isnothing(t_end)
    lo = isnothing(t_start) ? -Inf : t_start
    hi = isnothing(t_end) ? Inf : t_end
    tmin = nothing
    tmax = nothing
    for ev in events
        t = ev.time
        isfinite(t) || throw(ArgumentError("spike times must be finite"))
        c = get(counts, ev.neuron_id, 0)
        # Same bound-precision rule as the sorted-search train path, so both
        # overloads count identical windows.
        if !_population_below_lo(t, lo) && !_population_above_hi(t, hi)
            c += 1
        end
        counts[ev.neuron_id] = c
        need_extrema || continue
        tmin = isnothing(tmin) || isless(t, tmin) ? t : tmin
        tmax = isnothing(tmax) || isless(tmax, t) ? t : tmax
    end
    return counts, tmin, tmax, length(events)
end

function _population_snapshot(
    counts::AbstractDict{<:Integer,Int},
    duration::Float64,
    n_total_units::Int,
)::PopulationFeatureSnapshot
    n_observed = length(counts)
    n_active = 0
    total = 0
    for c in values(counts)
        total += c
        c > 0 && (n_active += 1)
    end

    rate = duration > 0 ? total / duration : 0.0
    mean_rate = 0.0
    frac = 0.0
    cmean = 0.0
    cvar = 0.0
    if n_total_units > 0
        mean_rate = rate / n_total_units
        frac = n_active / n_total_units
        cmean = total / n_total_units
        # uncorrected variance over n_total_units slots: absent declared units
        # contribute (0 - mean)^2 each.
        acc = (n_total_units - n_observed) * cmean * cmean
        for c in values(counts)
            d = c - cmean
            acc += d * d
        end
        cvar = acc / n_total_units
    end

    return PopulationFeatureSnapshot(
        duration,
        n_total_units,
        n_observed,
        n_active,
        total,
        rate,
        mean_rate,
        frac,
        cmean,
        cvar,
        cmean > 0 ? cvar / cmean : NaN,
    )
end

const _SpikePopulation = Union{AbstractVector{<:SpikeTrain},AbstractVector{<:SpikeEvent}}

"""
    population_spike_features(population; t_start=nothing, t_end=nothing,
                              n_total_units=nothing) -> PopulationFeatureSnapshot
    population_spike_features(stream::SpikeFeatureStream; n_total_units=nothing)
        -> Union{PopulationFeatureSnapshot,Nothing}

Summarize spike activity across a population of units into a
[`PopulationFeatureSnapshot`](@ref). `population` may be:

- a vector of [`SpikeTrain`](@ref)s — one entry per unit; empty trains are
  observed silent units, and trains sharing a `neuron_id` are merged;
- a vector of [`SpikeEvent`](@ref)s — a flat event list in any order; only the
  ids that appear are observed (events outside the window still mark their
  unit as observed);
- a single `SpikeTrain` — a one-unit population;
- a [`SpikeFeatureStream`](@ref) — aggregates the stream's current
  [`snapshot`](@ref) into population features for the active window, and
  returns `nothing` before the stream has accepted any event. Observed units
  are the snapshot's `neurons` rows (units with in-window spikes).

Neuron ids are opaque labels: they need not be dense, contiguous, positive or
small. All functions are pure — no global state is read or mutated.

# Window and denominator semantics

Spikes are counted inside the inclusive window `[t_start, t_end]`; a missing
bound is unbounded for counting. Each bound is applied in its own precision:
an `AbstractFloat` bound compares the `Float64` images of spike times — the
same domain [`spike_count`](@ref) uses — while an exact (integer or rational)
bound compares the stored values directly, preserving timestamp precision
beyond `Float64`. The `duration` used for rates is `t_end - t_start` with any
missing edge inferred from the population's earliest/latest spike — the same
convention as [`spike_density`](@ref) — so an inferred duration is `0` when
the population has fewer than two distinct edge times. A non-positive
`duration` yields rate fields of `0` while count fields are still reported.

`n_total_units` is the declared population size used as the denominator for
`mean_unit_rate`, `active_fraction`, `count_mean` and `count_variance`. It
defaults to the number of observed units and may be larger to account for
declared-but-absent (fully silent) units; a value smaller than the observed
unit count throws `ArgumentError`. `count_variance` is the uncorrected
variance over `n_total_units` slots, and `fano_factor` is `NaN` whenever
`count_mean == 0` (empty population or spikeless window).

# Examples

```julia
julia> pop = [SpikeTrain(1, [0.1, 0.2, 0.3]), SpikeTrain(7, [0.15]),
              SpikeTrain(1003, Float64[])];

julia> population_spike_features(pop; t_start = 0.0, t_end = 1.0)
PopulationFeatureSnapshot(1.0, 3, 3, 2, 4, 4.0, 1.3333333333333333, 0.6666666666666666, 1.3333333333333333, 1.5555555555555554, 1.1666666666666665)
```
"""
function population_spike_features(
    population::_SpikePopulation;
    t_start = nothing,
    t_end = nothing,
    n_total_units::Union{Nothing,Integer} = nothing,
)::PopulationFeatureSnapshot
    counts, tmin, tmax, n_spikes =
        _population_window_counts(population; t_start = t_start, t_end = t_end)
    duration = _population_duration(tmin, tmax, n_spikes, t_start, t_end)
    n_total = _resolve_n_total_units(n_total_units, length(counts))
    return _population_snapshot(counts, duration, n_total)
end

population_spike_features(train::SpikeTrain; kwargs...) =
    population_spike_features(SpikeTrain[train]; kwargs...)

function population_spike_features(
    stream::SpikeFeatureStream;
    n_total_units::Union{Nothing,Integer} = nothing,
)
    snap = snapshot(stream)
    snap === nothing && return nothing
    counts = Dict{Integer,Int}(row.neuron_id => row.features.count for row in snap.neurons)
    n_total = _resolve_n_total_units(n_total_units, length(counts))
    return _population_snapshot(counts, snap.t_end - snap.t_start, n_total)
end

"""
    unit_firing_rates(population; t_start=nothing, t_end=nothing) -> Vector
    unit_firing_rates(stream::SpikeFeatureStream) -> Union{Vector,Nothing}

Per-unit firing rates (spikes per unit time) over the observation window,
returned as a vector of `(neuron_id, spike_count, firing_rate)` named tuples
sorted by `neuron_id`. Sparse, negative or otherwise non-contiguous ids are
reported as-is; no dense id range is assumed.

Every observed unit appears exactly once, including units with zero spikes in
the window (`firing_rate == 0`). Declared-but-absent units have no id to list
and are therefore not included — use the `n_total_units` keyword of
[`population_spike_features`](@ref) to count them in aggregate metrics. For a
[`SpikeFeatureStream`](@ref) the rates are taken from the current
[`snapshot`](@ref)'s per-unit densities and the result is `nothing` before the
first accepted event.

`firing_rate` is `spike_count / duration` with the same window and inferred-
duration semantics as [`population_spike_features`](@ref); it is `0` whenever
the window `duration <= 0`.
"""
function unit_firing_rates(population::_SpikePopulation; t_start = nothing, t_end = nothing)
    counts, tmin, tmax, n_spikes =
        _population_window_counts(population; t_start = t_start, t_end = t_end)
    duration = _population_duration(tmin, tmax, n_spikes, t_start, t_end)
    rows = UnitRateRow[
        (neuron_id = id, spike_count = c, firing_rate = duration > 0 ? c / duration : 0.0) for (id, c) in counts
    ]
    return sort!(rows; by = row -> row.neuron_id)
end

unit_firing_rates(train::SpikeTrain; kwargs...) =
    unit_firing_rates(SpikeTrain[train]; kwargs...)

function unit_firing_rates(stream::SpikeFeatureStream)
    snap = snapshot(stream)
    snap === nothing && return nothing
    return sort!(
        UnitRateRow[
            (
                neuron_id = row.neuron_id,
                spike_count = row.features.count,
                firing_rate = row.features.density,
            ) for row in snap.neurons
        ];
        by = row -> row.neuron_id,
    )
end
