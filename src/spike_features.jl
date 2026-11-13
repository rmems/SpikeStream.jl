# SPDX-License-Identifier: MIT OR Apache-2.0

const WindowFeature = @NamedTuple{
    t_start::Float64,
    t_end::Float64,
    count::Int,
    density::Float64,
    isi_mean::Float64,
    isi_cv::Float64,
    burst_count::Int,
}

function _sorted_finite_times(spike_times::AbstractVector{<:Real})::Vector{Float64}
    n = length(spike_times)
    times = Vector{Float64}(undef, n)
    for i = 1:n
        t = Float64(spike_times[i])
        isfinite(t) || throw(ArgumentError("spike times must be finite"))
        times[i] = t
    end
    return sort!(times)
end

function _spike_count_sorted(
    times::AbstractVector{Float64};
    t_start = nothing,
    t_end = nothing,
)::Int
    if isnothing(t_start) && isnothing(t_end)
        return length(times)
    end

    start_t = isnothing(t_start) ? -Inf : Float64(t_start)
    end_t = isnothing(t_end) ? Inf : Float64(t_end)
    end_t < start_t && return 0

    left = searchsortedfirst(times, start_t)
    right = searchsortedlast(times, end_t)
    return max(0, right - left + 1)
end

function _spike_density_sorted(
    times::AbstractVector{Float64};
    t_start = nothing,
    t_end = nothing,
)::Float64
    inferred = isnothing(t_start) || isnothing(t_end)
    if inferred
        length(times) < 2 && return 0.0
        t_min = isnothing(t_start) ? first(times) : Float64(t_start)
        t_max = isnothing(t_end) ? last(times) : Float64(t_end)
    else
        t_min = Float64(t_start)
        t_max = Float64(t_end)
    end

    duration = t_max - t_min
    duration <= 0 && return 0.0

    n = _spike_count_sorted(times; t_start = t_min, t_end = t_max)
    return n / duration
end

function _n_windows(t0::Float64, t_end::Float64, step::Float64)::Int
    span = t_end - t0
    span <= 0 && return 0
    n = Int(floor(span / step))
    rem = span - n * step
    tol = eps(Float64) * max(abs(t0), abs(t_end), abs(step), 1.0) * 32
    rem > tol && (n += 1)
    return n
end

function _fp_ulp_band(a::Float64, b::Float64, others::Float64...)::Float64
    ref = max(maximum(abs, (a, b)), maximum(abs, others), 1.0)
    return 4 * eps(Float64) * ref
end

function _approx_equal(a::Float64, b::Float64, others::Float64...)::Bool
    return abs(a - b) <= _fp_ulp_band(a, b, others...)
end

function _window_end(
    t0::Float64,
    k::Int,
    step::Float64,
    window_size::Float64,
    t_end::Float64,
)::Float64
    ratio = window_size / step
    nearest = round(ratio)
    e = if abs(ratio - nearest) <= 32 * eps(Float64) * max(abs(ratio), 1.0)
        t0 + (k + Int(nearest)) * step
    else
        t0 + k * step + window_size
    end
    snap_tol = _fp_ulp_band(e, t_end, t0, step, window_size)
    return (e >= t_end || abs(e - t_end) <= snap_tol) ? t_end : e
end

function _start_index(t::Float64, t0::Float64, step::Float64)::Int
    q = (t - t0) / step
    nearest = round(q)
    scale = max(abs(t), abs(t0), abs(step), 1.0) / step
    if abs(q - nearest) <= 32 * eps(Float64) * max(abs(q), scale, 1.0)
        return Int(nearest)
    end
    return Int(floor(q))
end

function _window_span_steps(window_size::Float64, step::Float64)
    ratio = window_size / step
    nearest = round(ratio)
    if abs(ratio - nearest) <= 32 * eps(Float64) * max(abs(ratio), 1.0)
        return Int(nearest)
    end
    return nothing
end

function _spike_window_index(
    t::Float64,
    t0::Float64,
    step::Float64,
    n_win::Int,
    t_end::Float64,
    include_right_edge::Bool,
)::Int
    if t < t0
        return -1
    end
    if include_right_edge && (t == t_end || _approx_equal(t, t_end, t0, step))
        return n_win - 1
    end
    if t >= t_end
        return n_win
    end
    return min(_start_index(t, t0, step), n_win - 1)
end

function _spike_in_window(
    t::Float64,
    k::Int,
    t0::Float64,
    step::Float64,
    n_win::Int,
    t_end::Float64,
    cur::Float64,
    nxt::Float64,
    window_size::Float64,
    include_right_edge::Bool,
    span_steps::Union{Int,Nothing},
)::Bool
    idx = _spike_window_index(t, t0, step, n_win, t_end, include_right_edge)
    if !isnothing(span_steps)
        idx < n_win || return false
        return k <= idx < k + span_steps
    end
    idx < k && return false
    closed = include_right_edge && (nxt >= t_end || _approx_equal(nxt, t_end, t0, step))
    if closed
        return t <= nxt ||
               _approx_equal(t, nxt, cur, step) ||
               _approx_equal(t, t_end, t0, step)
    end
    hi = min(cur + window_size, t_end)
    span = hi - cur
    limit = span > window_size && idx > k ? prevfloat(hi) : hi
    left = t >= cur || _approx_equal(t, cur, t0, step)
    return left && t < limit
end

"""
    spike_count(spike_times; t_start=nothing, t_end=nothing) -> Int

Count spikes in `spike_times`, optionally restricted to `[t_start, t_end]`.

`spike_times` should be finite numeric timestamps in seconds (or any consistent unit).
When no window is provided, all spikes are counted.
"""
function spike_count(
    spike_times::AbstractVector{<:Real};
    t_start = nothing,
    t_end = nothing,
)::Int
    times = _sorted_finite_times(spike_times)
    return _spike_count_sorted(times; t_start = t_start, t_end = t_end)
end

"""
    spike_density(spike_times; t_start=nothing, t_end=nothing) -> Float64

Compute spike density as `spike_count / window_duration`.

When both `t_start` and `t_end` are given and the duration is positive,
density is `n / duration` even for 0 or 1 spike. If either bound is omitted,
the missing edge is taken from the first or last spike; that inferred
duration is undefined for fewer than 2 spikes and returns `0.0`. Returns
`0.0` when the window duration is non-positive.
"""
function spike_density(
    spike_times::AbstractVector{<:Real};
    t_start = nothing,
    t_end = nothing,
)::Float64
    times = _sorted_finite_times(spike_times)
    return _spike_density_sorted(times; t_start = t_start, t_end = t_end)
end

"""
    isi_stats(spike_times) -> NamedTuple

Compute inter-spike-interval statistics.

Returns `(mean, std, min, max, cv)` where CV is coefficient of variation `std/mean`.
All values are non-negative. For fewer than 2 spikes, returns zeros.
"""
function isi_stats(spike_times::AbstractVector{<:Real})
    times = _sorted_finite_times(spike_times)
    return isi_stats_sorted(times)
end

function isi_stats_sorted(sorted_spike_times::AbstractVector{Float64})
    length(sorted_spike_times) < 2 &&
        return (mean = 0.0, std = 0.0, min = 0.0, max = 0.0, cv = 0.0)

    isis = diff(sorted_spike_times)
    μ = mean(isis)
    σ = length(isis) > 1 ? std(isis) : 0.0
    min_isi = minimum(isis)
    max_isi = maximum(isis)
    cv = μ > 0 ? σ / μ : 0.0

    return (mean = μ, std = σ, min = min_isi, max = max_isi, cv = cv)
end

"""
    detect_bursts(spike_times; max_isi=0.02, min_spikes=3) -> Vector{UnitRange{Int}}

Detect bursts as contiguous runs where each inter-spike interval is `<= max_isi`.
Returns index ranges into the sorted spike-time sequence.
"""
function detect_bursts(
    spike_times::AbstractVector{<:Real};
    max_isi::Real = 0.02,
    min_spikes::Int = 3,
)
    times = _sorted_finite_times(spike_times)
    return detect_bursts_sorted(times; max_isi = max_isi, min_spikes = min_spikes)
end

function detect_bursts_sorted(
    sorted_spike_times::AbstractVector{Float64};
    max_isi::Real = 0.02,
    min_spikes::Int = 3,
)
    n = length(sorted_spike_times)
    n == 0 && return UnitRange{Int}[]
    n < min_spikes && return UnitRange{Int}[]

    isis = diff(sorted_spike_times)
    bursts = UnitRange{Int}[]
    run_start = 1
    run_len = 1

    for (i, isi) in enumerate(isis)
        if isi <= max_isi
            run_len += 1
        else
            if run_len >= min_spikes
                push!(bursts, run_start:(run_start+run_len-1))
            end
            run_start = i + 1
            run_len = 1
        end
    end

    if run_len >= min_spikes
        push!(bursts, run_start:(run_start+run_len-1))
    end

    return bursts
end

"""
    windowed_spike_features(spike_times; window_size, step=window_size, t_start=nothing, t_end=nothing)

Extract spike features per window. Returns a vector of named tuples with:
`(t_start, t_end, count, density, isi_mean, isi_cv, burst_count)`.
"""
function windowed_spike_features(
    spike_times::AbstractVector{<:Real};
    window_size::Real,
    step::Real = window_size,
    t_start = nothing,
    t_end = nothing,
)
    window_size <= 0 && throw(ArgumentError("window_size must be > 0"))
    step <= 0 && throw(ArgumentError("step must be > 0"))

    times = _sorted_finite_times(spike_times)

    include_right_edge = false
    if isempty(times)
        base_start = isnothing(t_start) ? 0.0 : Float64(t_start)
        base_end = isnothing(t_end) ? 0.0 : Float64(t_end)
    else
        base_start = isnothing(t_start) ? first(times) : Float64(t_start)
        if isnothing(t_end)
            base_end = last(times)
            include_right_edge = true
        else
            base_end = Float64(t_end)
        end
    end

    step_f = Float64(step)
    n_win = _n_windows(base_start, base_end, step_f)
    n_win == 0 && return WindowFeature[]

    features = Vector{WindowFeature}(undef, n_win)
    left = 1
    n = length(times)
    ws = Float64(window_size)
    span_steps = _window_span_steps(ws, step_f)

    for k = 0:(n_win-1)
        cur = base_start + k * step_f
        nxt = _window_end(base_start, k, step_f, ws, base_end)

        while left <= n &&
              !_spike_in_window(
            times[left],
            k,
            base_start,
            step_f,
            n_win,
            base_end,
            cur,
            nxt,
            ws,
            include_right_edge,
            span_steps,
        )
            idx = _spike_window_index(
                times[left],
                base_start,
                step_f,
                n_win,
                base_end,
                include_right_edge,
            )
            beyond = isnothing(span_steps) ? 1 : span_steps
            idx >= k + beyond && break
            left += 1
        end
        right = left - 1
        while right < n && _spike_in_window(
            times[right+1],
            k,
            base_start,
            step_f,
            n_win,
            base_end,
            cur,
            nxt,
            ws,
            include_right_edge,
            span_steps,
        )
            right += 1
        end

        local_count = max(0, right - left + 1)
        local_times = local_count > 0 ? (@view times[left:right]) : Float64[]

        d = nxt > cur ? local_count / (nxt - cur) : 0.0
        stats = isi_stats_sorted(local_times)
        b = length(detect_bursts_sorted(local_times))

        features[k+1] = WindowFeature((cur, nxt, local_count, d, stats.mean, stats.cv, b))
    end

    return features
end

"""
    SpikeFeatureSnapshot

Canonical raw spike features for one observation window.

Fields are physical/statistical quantities (not scaled to `[0, 1]`):

- `count` — spike count (`>= 0`)
- `density` — spikes per unit time (`>= 0`)
- `isi_mean` — mean inter-spike interval (`>= 0`)
- `isi_cv` — coefficient of variation of ISIs (`>= 0`)
- `burst_count` — number of detected bursts (`>= 0`)

Downstream normalization is separate: use [`SpikeFeatureNormalizer`](@ref) with
[`transform`](@ref). Batch helpers and future streaming snapshots should share
this shape so one normalizer configuration applies to both.
"""
struct SpikeFeatureSnapshot
    count::Int
    density::Float64
    isi_mean::Float64
    isi_cv::Float64
    burst_count::Int
end

"""
    spike_features(spike_times; t_start=nothing, t_end=nothing, max_isi=0.02, min_spikes=3) -> SpikeFeatureSnapshot

Extract the canonical raw feature snapshot for `spike_times` (optionally windowed).

Does not apply scaling or clipping; pass the result to [`transform`](@ref) with an
explicit [`SpikeFeatureNormalizer`](@ref) when a bounded vector is needed.
"""
function spike_features(
    spike_times::AbstractVector{<:Real};
    t_start = nothing,
    t_end = nothing,
    max_isi::Real = 0.02,
    min_spikes::Integer = 3,
)::SpikeFeatureSnapshot
    times = _sorted_finite_times(spike_times)
    start_t = isnothing(t_start) ? -Inf : Float64(t_start)
    end_t = isnothing(t_end) ? Inf : Float64(t_end)
    left = searchsortedfirst(times, start_t)
    right = searchsortedlast(times, end_t)
    window_times = left <= right ? times[left:right] : Float64[]

    c = _spike_count_sorted(times; t_start = t_start, t_end = t_end)
    d = _spike_density_sorted(times; t_start = t_start, t_end = t_end)
    stats = isi_stats_sorted(window_times)
    b = length(
        detect_bursts_sorted(window_times; max_isi = max_isi, min_spikes = Int(min_spikes)),
    )

    return SpikeFeatureSnapshot(c, d, stats.mean, stats.cv, b)
end

"""
    SpikeFeatureNormalizer(; max_density=1000.0, max_isi_cv=2.0, max_count=1.0,
                           count_mode=:presence, clip=true)

Explicit, serializable scaling policy from [`SpikeFeatureSnapshot`](@ref) to a
length-4 feature vector.

# Scaling

| Component | Policy |
| --- | --- |
| count | `:presence` → `1.0` if `count > 0` else `0.0`; `:scaled` → `count / max_count` |
| density | `density / max_density` |
| isi_cv | `isi_cv / max_isi_cv` |
| burst | `burst_count / max(count, 1)` (relative to the snapshot's spike count) |

When `clip=true` (default), each component is clamped to `[0, 1]`. Scales must be
positive and finite. No training-data statistics are estimated.

The default constructor matches the historical [`normalized_feature_vector`](@ref)
behavior (`count_mode=:presence`, `max_isi_cv=2.0`, `clip=true`).
"""
struct SpikeFeatureNormalizer
    max_density::Float64
    max_isi_cv::Float64
    max_count::Float64
    count_mode::Symbol
    clip::Bool

    function SpikeFeatureNormalizer(max_density, max_isi_cv, max_count, count_mode, clip)
        md = Float64(max_density)
        mc = Float64(max_isi_cv)
        mcount = Float64(max_count)
        mode = convert(Symbol, count_mode)
        isfinite(md) && md > 0 ||
            throw(ArgumentError("max_density must be a positive finite value"))
        isfinite(mc) && mc > 0 ||
            throw(ArgumentError("max_isi_cv must be a positive finite value"))
        isfinite(mcount) && mcount > 0 ||
            throw(ArgumentError("max_count must be a positive finite value"))
        mode === :presence ||
            mode === :scaled ||
            throw(
                ArgumentError("count_mode must be :presence or :scaled, got $(repr(mode))"),
            )
        return new(md, mc, mcount, mode, Bool(clip))
    end
end

function SpikeFeatureNormalizer(;
    max_density::Real = 1000.0,
    max_isi_cv::Real = 2.0,
    max_count::Real = 1.0,
    count_mode::Symbol = :presence,
    clip::Bool = true,
)
    return SpikeFeatureNormalizer(max_density, max_isi_cv, max_count, count_mode, clip)
end

"""
    transform(normalizer::SpikeFeatureNormalizer, features::SpikeFeatureSnapshot) -> Vector{Float64}

Apply `normalizer` to a raw snapshot, returning
`[count_norm, density_norm, isi_cv_norm, burst_norm]`.
"""
function transform(
    normalizer::SpikeFeatureNormalizer,
    features::SpikeFeatureSnapshot,
)::Vector{Float64}
    count_norm = if normalizer.count_mode === :presence
        features.count > 0 ? 1.0 : 0.0
    else
        features.count / normalizer.max_count
    end
    density_norm = features.density / normalizer.max_density
    isi_cv_norm = features.isi_cv / normalizer.max_isi_cv
    burst_norm = features.burst_count / max(features.count, 1)

    vec = [count_norm, density_norm, isi_cv_norm, burst_norm]
    if normalizer.clip
        @inbounds for i in eachindex(vec)
            vec[i] = clamp(vec[i], 0.0, 1.0)
        end
    end
    return vec
end

"""
    normalized_feature_vector(spike_times; t_start=nothing, t_end=nothing,
                              max_density=1000.0, normalizer=nothing,
                              max_isi=0.02, min_spikes=3) -> Vector{Float64}

Compatibility helper: extract a [`SpikeFeatureSnapshot`](@ref) then
[`transform`](@ref) it.

If `normalizer` is omitted, uses `SpikeFeatureNormalizer(; max_density=max_density)`
(historical default scales). Prefer constructing a [`SpikeFeatureNormalizer`](@ref)
explicitly when scales should be shared across batch and streaming callers; when
`normalizer` is provided, `max_density` is ignored.
"""
function normalized_feature_vector(
    spike_times::AbstractVector{<:Real};
    t_start = nothing,
    t_end = nothing,
    max_density::Real = 1000.0,
    normalizer::Union{Nothing,SpikeFeatureNormalizer} = nothing,
    max_isi::Real = 0.02,
    min_spikes::Integer = 3,
)::Vector{Float64}
    snap = spike_features(
        spike_times;
        t_start = t_start,
        t_end = t_end,
        max_isi = max_isi,
        min_spikes = min_spikes,
    )
    n =
        isnothing(normalizer) ? SpikeFeatureNormalizer(; max_density = max_density) :
        normalizer
    return transform(n, snap)
end
