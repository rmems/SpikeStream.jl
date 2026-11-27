# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using Random
using SpikeStream

const CONVERSION_COUNT = Ref(0)

struct ConversionProbe <: Real
    value::Float64
end
Base.Float64(x::ConversionProbe) = (CONVERSION_COUNT[] += 1; x.value)
Base.isfinite(x::ConversionProbe) = isfinite(x.value)

const TICK_BOUND_CMPS = Ref(0)

# A finite Real timestamp with exact comparisons and a lossless BigFloat
# image but no Float64 conversion: omitted or exact window bounds must never
# reach for Float64, and bound<->tick isless calls are counted so a binary
# search is distinguishable from a full scan.
struct CmpTick <: Real
    value::Int
end
Base.isfinite(::CmpTick) = true
Base.isless(a::CmpTick, b::CmpTick) = isless(a.value, b.value)
Base.isless(a::Integer, b::CmpTick) = (TICK_BOUND_CMPS[] += 1; isless(a, b.value))
Base.isless(a::CmpTick, b::Integer) = (TICK_BOUND_CMPS[] += 1; isless(a.value, b))
Base.BigFloat(t::CmpTick) = BigFloat(t.value)

@testset "SpikeStream" begin

    @testset "Package loads" begin
        @test @isdefined(SpikeStream)
        @test SpikeStream isa Module
        @test isdefined(SpikeStream, :spike_count)
        @test isdefined(SpikeStream, :spike_density)
        @test isdefined(SpikeStream, :isi_stats)
        @test isdefined(SpikeStream, :detect_bursts)
        @test isdefined(SpikeStream, :windowed_spike_features)
        @test isdefined(SpikeStream, :spike_features)
        @test isdefined(SpikeStream, :SpikeFeatureSnapshot)
        @test isdefined(SpikeStream, :SpikeFeatureNormalizer)
        @test isdefined(SpikeStream, :transform)
        @test isdefined(SpikeStream, :normalized_feature_vector)
        for name in (
            :SpikeEvent,
            :SpikeTrain,
            :spike_times,
            :neuron_id,
            :spike_events,
            :spike_trains,
            :SpikeFeatureStream,
            :snapshot,
            :PopulationFeatureSnapshot,
            :population_spike_features,
            :unit_firing_rates,
        )
            @test Base.isexported(SpikeStream, name)
            @test haskey(Base.Docs.meta(SpikeStream), Base.Docs.Binding(SpikeStream, name))
        end
    end

    @testset "spike_count + spike_density" begin
        spikes = [0.1, 0.2, 0.25, 0.9]
        @test spike_count(spikes) == 4
        @test spike_count(spikes; t_start = 0.15, t_end = 0.3) == 2
        @test spike_density(spikes; t_start = 0.0, t_end = 1.0) == 4.0
        @test spike_density([0.1]) == 0.0

        # partial bounds should be respected
        @test spike_density(spikes; t_start = 0.2) ≈ 3 / (0.9 - 0.2)
        @test spike_density(spikes; t_end = 0.25) ≈ 3 / (0.25 - 0.1)

        # explicit observation window is the duration, even for a single spike
        @test spike_density([0.3]; t_start = 0.0, t_end = 1.0) == 1.0
        @test spike_density(Float64[]; t_start = 0.0, t_end = 1.0) == 0.0

        # one inferred edge still needs two spikes (Blocks: [0.999]; t_end=1.0)
        @test spike_density([0.999]; t_end = 1.0) == 0.0
        @test spike_density([0.5]; t_start = 0.0) == 0.0
        @test normalized_feature_vector([0.999]; t_end = 1.0)[2] == 0.0

        # inferred duration from first/last spike when n ≥ 2
        @test spike_density([0.0, 1.0]) == 2.0
        @test spike_count([0.1, 0.2]; t_start = 1.0, t_end = 0.0) == 0
        @test spike_density([0.5]; t_start = 1.0, t_end = 1.0) == 0.0
        @test spike_density(Float64[]; t_start = 0.2) == 0.0
        @test spike_density(Float64[]; t_end = 0.2) == 0.0
    end

    @testset "reject non-finite spike times" begin
        for bad in (NaN, Inf, -Inf)
            times = [0.1, bad, 0.3]
            @test_throws ArgumentError spike_count(times)
            @test_throws ArgumentError spike_density(times; t_start = 0.0, t_end = 1.0)
            @test_throws ArgumentError isi_stats(times)
            @test_throws ArgumentError detect_bursts(times)
            @test_throws ArgumentError windowed_spike_features(
                times;
                window_size = 0.1,
                step = 0.1,
                t_start = 0.0,
                t_end = 1.0,
            )
            @test_throws ArgumentError spike_features(times)
            @test_throws ArgumentError normalized_feature_vector(times)
        end
    end

    @testset "ISI stats" begin
        spikes = [0.1, 0.2, 0.4, 0.7]
        stats = isi_stats(spikes)
        @test stats.mean ≈ 0.2
        @test stats.min ≈ 0.1
        @test stats.max ≈ 0.3
        @test stats.cv ≥ 0

        empty_stats = isi_stats([0.1])
        @test empty_stats.mean == 0.0

        two_spike = isi_stats([0.1, 0.5])
        @test two_spike.mean ≈ 0.4
        @test two_spike.std == 0.0
        @test two_spike.cv == 0.0
    end

    @testset "burst detection" begin
        spikes = [0.0, 0.005, 0.009, 0.1, 0.2]
        bursts = detect_bursts(spikes; max_isi = 0.01, min_spikes = 3)
        @test length(bursts) == 1
        @test first(bursts) == 1:3
    end

    @testset "windowed extraction" begin
        spikes = [0.01, 0.02, 0.06, 0.07]
        feats = windowed_spike_features(
            spikes;
            window_size = 0.05,
            step = 0.05,
            t_start = 0.0,
            t_end = 0.1,
        )
        @test length(feats) == 2
        @test feats[1].count == 2
        @test feats[2].count == 2
        @test all(f -> f.density ≥ 0, feats)

        # auto t_end includes the final spike
        auto_feats = windowed_spike_features([0.0, 0.5, 1.0]; window_size = 0.5, step = 0.5)
        @test sum(f.count for f in auto_feats) == 3

        # integer-k starts: no float-drift extra sliver, 0.3 bins into [0.3, 0.4)
        grid = windowed_spike_features(
            [0.3];
            window_size = 0.1,
            step = 0.1,
            t_start = 0.0,
            t_end = 10.0,
        )
        @test length(grid) == 100
        @test isconcretetype(eltype(grid))
        @test grid[end].t_end - grid[end].t_start ≈ 0.1
        @test grid[3].count == 0
        @test grid[4].count == 1
        @test grid[4].t_start ≈ 0.3
        @test grid[4].t_end ≈ 0.4
    end

    @testset "window edges, slivers, and errors" begin
        @test_throws ArgumentError windowed_spike_features([0.1]; window_size = 0.0)
        @test_throws ArgumentError windowed_spike_features([0.1]; window_size = -0.1)
        @test_throws ArgumentError windowed_spike_features(
            [0.1];
            window_size = 0.1,
            step = 0.0,
        )

        empty_auto = windowed_spike_features(Float64[]; window_size = 0.1)
        @test isempty(empty_auto)
        @test isconcretetype(eltype(empty_auto))

        empty_span = windowed_spike_features(
            Float64[];
            window_size = 0.1,
            t_start = 0.0,
            t_end = 1.0,
        )
        @test length(empty_span) == 10

        inverted =
            windowed_spike_features([0.2]; window_size = 0.1, t_start = 1.0, t_end = 0.0)
        @test isempty(inverted)

        # real remainder is an extra short window; float slivers are not
        remainder = windowed_spike_features(
            Float64[];
            window_size = 0.1,
            step = 0.1,
            t_start = 0.0,
            t_end = 1.05,
        )
        @test length(remainder) == 11
        @test remainder[end].t_end - remainder[end].t_start ≈ 0.05

        sliver_end = 10.0 + 2e-14
        sliver = windowed_spike_features(
            Float64[];
            window_size = 0.1,
            step = 0.1,
            t_start = 0.0,
            t_end = sliver_end,
        )
        @test length(sliver) == 100

        # spikes before t_start are skipped (index -1)
        before = windowed_spike_features(
            [-0.4, 0.05, 0.55];
            window_size = 0.1,
            step = 0.1,
            t_start = 0.0,
            t_end = 0.3,
        )
        @test length(before) == 3
        @test before[1].count == 1
        @test sum(f.count for f in before) == 1

        # non-integer window/step uses float edges instead of integer span
        uneven = windowed_spike_features(
            [0.01, 0.12, 0.28];
            window_size = 0.15,
            step = 0.1,
            t_start = 0.0,
            t_end = 0.4,
        )
        @test length(uneven) == 4
        @test uneven[1].count == 2
        @test uneven[2].count == 1
        @test uneven[3].count == 1
        @test sum(f.count for f in uneven) >= 3

        # implicit t_end, non-integer ratio, last spike on the closed right edge
        closed_last = windowed_spike_features([0.0, 0.22]; window_size = 0.15, step = 0.1)
        @test sum(f.count for f in closed_last) >= 2
        @test closed_last[end].count >= 1

        after_end = windowed_spike_features(
            [0.05, 1.0];
            window_size = 0.1,
            t_start = 0.0,
            t_end = 0.5,
        )
        @test sum(f.count for f in after_end) == 1

        empty_nfv = normalized_feature_vector([0.5]; t_start = 1.0, t_end = 2.0)
        @test empty_nfv[1] == 0.0
        @test all(empty_nfv .== 0.0)

        overlap = windowed_spike_features(
            [0.5, 1.5, 5.0, 6.0];
            window_size = 2.0,
            step = 1.0,
            t_start = 0.0,
            t_end = 2.0,
        )
        @test length(overlap) == 2
        @test overlap[1].count == 2
        @test overlap[2].count == 1
        @test overlap[2].t_start ≈ 1.0
        @test overlap[2].t_end ≈ 2.0

        implicit_overlap =
            windowed_spike_features([0.0, 0.5, 1.0]; window_size = 0.7, step = 0.3)
        @test implicit_overlap[1].count == 2
        @test implicit_overlap[2].count == 2
        @test implicit_overlap[3].count == 1
        @test sum(f.count for f in implicit_overlap) >= 3

        # 0.3 snaps to 3*0.1; must stay in the last in-range window, not n_win
        snapped = windowed_spike_features(
            [0.3];
            window_size = 0.1,
            t_start = 0.0,
            t_end = 3 * 0.1,
        )
        @test length(snapped) == 3
        @test sum(f.count for f in snapped) == 1
        @test snapped[end].count == 1
        @test spike_count([0.3]; t_start = 0.0, t_end = 3 * 0.1) == 1

        # non-integer window/step: 0.3 still belongs in the window at 3*0.1
        float_snap = windowed_spike_features(
            [0.3];
            window_size = 0.15,
            step = 0.1,
            t_start = 0.0,
            t_end = 0.6,
        )
        @test length(float_snap) == 6
        @test float_snap[4].t_start ≈ 0.3
        @test float_snap[4].count == 1
        @test float_snap[3].count == 1

        # subtraction error at large t0 is bigger than 32*eps*|q|
        offset = windowed_spike_features(
            [100.3];
            window_size = 0.1,
            t_start = 100.0,
            t_end = 101.0,
        )
        @test any(f.t_start ≈ 100.3 && f.count == 1 for f in offset)
        @test all(!(f.t_start ≈ 100.2 && f.count == 1) for f in offset)

        offset_overlap = windowed_spike_features(
            [16.1, 16.4];
            window_size = 0.7,
            step = 0.1,
            t_end = 16.5,
        )
        @test any(f.t_start ≈ 16.4 && f.count == 1 for f in offset_overlap)

        # non-integer window/step: implicit last spike must not drop when nxt < t_end by 1 ulp
        drift_last = windowed_spike_features([0.0, 0.45]; window_size = 0.15, step = 0.3)
        @test length(drift_last) == 2
        @test drift_last[end].count == 1
        @test drift_last[end].t_end ≈ 0.45
        @test sum(f.count for f in drift_last) == 2

        drift_off = windowed_spike_features([0.59, 1.04]; window_size = 0.15, step = 0.3)
        @test drift_off[end].count == 1

        # non-integer path: spike exactly on a nominal end stays out of [start, end)
        over_end = windowed_spike_features(
            [1.4];
            window_size = 0.3,
            step = 0.2,
            t_start = 0.5,
            t_end = 2.0,
        )
        @test sum(f.count for f in over_end) == 1
        @test all(!(f.t_start ≈ 1.1 && f.count == 1) for f in over_end)
        @test any(f.t_start ≈ 1.3 && f.count == 1 for f in over_end)

        # 32-ulp equality is wider than window-end rounding at large t0
        epoch = 1.7e9
        epoch_feats = windowed_spike_features(
            [epoch, epoch + 0.15 - 5e-6, epoch + 0.45];
            window_size = 0.15,
            step = 0.3,
        )
        @test epoch_feats[1].count == 2
        @test epoch_feats[2].count == 1

        mid = 1000.0
        mid_feats = windowed_spike_features(
            [mid, mid + 0.15 - 1e-12, mid + 0.45];
            window_size = 0.15,
            step = 0.3,
        )
        @test mid_feats[1].count == 2
        @test mid_feats[2].count == 1

        # prevfloat(window end) is interior; must not be excluded by end tolerance
        ulp_in = windowed_spike_features(
            [prevfloat(0.15)];
            window_size = 0.15,
            step = 0.3,
            t_start = 0.0,
            t_end = 0.6,
        )
        @test sum(f.count for f in ulp_in) == 1
        @test ulp_in[1].count == 1

        # span rounding above window_size must not drop the last-candidate spike
        gap_edge = windowed_spike_features(
            [0.3];
            window_size = 0.2,
            step = 0.3,
            t_start = 0.1,
            t_end = 1.0,
        )
        @test sum(f.count for f in gap_edge) == 1
        @test gap_edge[1].count == 1

        # membership uses the snapped end: prevfloat(t_end) stays in the final window
        snapped_end = windowed_spike_features(
            [prevfloat(0.45)];
            window_size = 0.15,
            step = 0.3,
            t_start = 0.0,
            t_end = 0.45,
        )
        @test sum(f.count for f in snapped_end) == 1
        @test snapped_end[end].count == 1
    end

    @testset "normalized vector" begin
        spikes = [0.0, 0.01, 0.02, 0.2]
        vec = normalized_feature_vector(
            spikes;
            t_start = 0.0,
            t_end = 1.0,
            max_density = 10.0,
        )
        @test length(vec) == 4
        @test all(0.0 .≤ vec .≤ 1.0)

        vec_window = normalized_feature_vector(
            spikes;
            t_start = 0.0,
            t_end = 0.03,
            max_density = 100.0,
        )
        @test vec_window[4] ≈ 1 / 3

        # one conversion/sort of the input (not once per helper)
        CONVERSION_COUNT[] = 0
        probed = ConversionProbe[ConversionProbe(t) for t in (0.4, 0.0, 0.2, 0.1)]
        got = normalized_feature_vector(probed; t_start = 0.0, t_end = 1.0)
        @test got ==
              normalized_feature_vector([0.4, 0.0, 0.2, 0.1]; t_start = 0.0, t_end = 1.0)
        @test CONVERSION_COUNT[] == length(probed)
    end

    @testset "spike features + configurable normalizer" begin
        spikes = [0.0, 0.01, 0.02, 0.2]
        snap = spike_features(spikes; t_start = 0.0, t_end = 1.0)
        @test snap isa SpikeFeatureSnapshot
        @test snap.count == 4
        @test snap.density == 4.0
        @test snap.isi_mean ≥ 0
        @test snap.isi_cv ≥ 0
        @test snap.burst_count ≥ 0

        empty_snap = spike_features(Float64[]; t_start = 0.0, t_end = 1.0)
        @test empty_snap == SpikeFeatureSnapshot(0, 0.0, 0.0, 0.0, 0)

        default_n = SpikeFeatureNormalizer(; max_density = 10.0)
        @test transform(default_n, snap) == normalized_feature_vector(
            spikes;
            t_start = 0.0,
            t_end = 1.0,
            max_density = 10.0,
        )
        @test transform(default_n, empty_snap) == [0.0, 0.0, 0.0, 0.0]

        # large values clip to 1 under default policy
        large = SpikeFeatureSnapshot(10, 5000.0, 0.1, 10.0, 10)
        clipped = transform(SpikeFeatureNormalizer(; max_density = 1000.0), large)
        @test clipped == [1.0, 1.0, 1.0, 1.0]

        # clip=false leaves unscaled ratios above 1
        unclipped = transform(
            SpikeFeatureNormalizer(; max_density = 1000.0, max_isi_cv = 2.0, clip = false),
            large,
        )
        @test unclipped[1] == 1.0
        @test unclipped[2] ≈ 5.0
        @test unclipped[3] ≈ 5.0
        @test unclipped[4] == 1.0

        # custom scales + scaled count mode
        scaled_n = SpikeFeatureNormalizer(;
            max_density = 8.0,
            max_isi_cv = 4.0,
            max_count = 8.0,
            count_mode = :scaled,
            clip = true,
        )
        scaled_vec = transform(scaled_n, snap)
        @test scaled_vec[1] ≈ 4 / 8
        @test scaled_vec[2] ≈ clamp(snap.density / 8.0, 0.0, 1.0)
        @test scaled_vec[3] ≈ clamp(snap.isi_cv / 4.0, 0.0, 1.0)
        @test scaled_vec[4] ≈ clamp(snap.burst_count / max(snap.count, 1), 0.0, 1.0)

        # reproducibility: reconstructing the same kwargs yields the same transform
        n1 = SpikeFeatureNormalizer(;
            max_density = 200.0,
            max_isi_cv = 1.5,
            max_count = 5.0,
            count_mode = :scaled,
            clip = false,
        )
        n2 = SpikeFeatureNormalizer(;
            max_density = 200.0,
            max_isi_cv = 1.5,
            max_count = 5.0,
            count_mode = :scaled,
            clip = false,
        )
        @test n1 == n2
        @test transform(n1, snap) == transform(n2, snap)

        # compatibility: explicit normalizer kwarg on normalized_feature_vector
        @test normalized_feature_vector(
            spikes;
            t_start = 0.0,
            t_end = 1.0,
            normalizer = scaled_n,
        ) == scaled_vec

        @test_throws ArgumentError SpikeFeatureNormalizer(; count_mode = :bogus)
        @test_throws ArgumentError SpikeFeatureNormalizer(; max_density = 0.0)
        @test_throws ArgumentError SpikeFeatureNormalizer(; max_isi_cv = -1.0)
        @test_throws ArgumentError SpikeFeatureNormalizer(; max_count = 0.0)
        @test_throws ArgumentError SpikeFeatureNormalizer(; max_density = Inf)
        @test_throws ArgumentError SpikeFeatureNormalizer(; max_density = NaN)
        @test_throws ArgumentError SpikeFeatureNormalizer(0.0, 2.0, 1.0, :presence, true)
        @test_throws ArgumentError SpikeFeatureNormalizer(Inf, 2.0, 1.0, :bogus, true)

        # non-Int Integer min_spikes and degenerate burst inputs
        @test spike_features(spikes; min_spikes = Int32(3)).burst_count ==
              spike_features(spikes).burst_count
        @test spike_features(Float64[]; min_spikes = 0).burst_count == 0
        @test isempty(detect_bursts(Float64[]; min_spikes = 0))

        # docstring coverage for new exports
        @test !isempty(string(@doc SpikeFeatureSnapshot))
        @test !isempty(string(@doc spike_features))
        @test !isempty(string(@doc SpikeFeatureNormalizer))
        @test !isempty(string(@doc transform))
    end

    @testset "SpikeEvent" begin
        ev = SpikeEvent(0.25, 3)
        @test ev isa SpikeEvent{Float64,Int}
        @test ev.time == 0.25
        @test neuron_id(ev) == 3

        @test SpikeEvent(Float32(0.5), Int16(2)) isa SpikeEvent{Float32,Int16}
        @test SpikeEvent(7, UInt8(1)) isa SpikeEvent{Int,UInt8}
        @test SpikeEvent(1 // 3, 0) isa SpikeEvent{Rational{Int},Int}
        @test SpikeEvent{Float64,Int32}(1, 2) isa SpikeEvent{Float64,Int32}
        @test neuron_id(SpikeEvent(0.0, -4)) == -4

        for bad in (NaN, Inf, -Inf, NaN32)
            @test_throws ArgumentError SpikeEvent(bad, 1)
            @test_throws ArgumentError SpikeEvent{Float64,Int}(bad, 1)
        end

        @test isless(SpikeEvent(0.1, 9), SpikeEvent(0.2, 1))
        @test isless(SpikeEvent(0.1, 7), SpikeEvent(0.1, 9))
        @test !isless(SpikeEvent(0.1, 7), SpikeEvent(0.1, 7))
        @test SpikeEvent(0.1, 7) == SpikeEvent(0.1, 7)
        @test sort([SpikeEvent(0.2, 1), SpikeEvent(0.1, 2), SpikeEvent(0.1, 1)]) ==
              [SpikeEvent(0.1, 1), SpikeEvent(0.1, 2), SpikeEvent(0.2, 1)]

        mixed = [SpikeEvent(Float32(0.5), Int16(1)), SpikeEvent(0.25, 2)]
        @test eltype(mixed) == SpikeEvent{Float64,Int}
        @test convert(SpikeEvent{Float64,Int}, SpikeEvent(1, Int8(2))) == SpikeEvent(1.0, 2)
        same = SpikeEvent(1.0, 2)
        @test convert(SpikeEvent{Float64,Int}, same) === same
        # Narrowing to a type where the value overflows must not yield ±Inf.
        @test_throws ArgumentError SpikeEvent{Float32,Int}(1e300, 1)
        @test_throws ArgumentError SpikeEvent{Float16,Int}(1e5, 1)
        @test_throws ArgumentError convert(SpikeEvent{Float32,Int}, SpikeEvent(1e300, 1))
        # NaN/±Inf raise ArgumentError even when T cannot represent them
        for bad in (NaN, Inf, -Inf)
            @test_throws ArgumentError SpikeEvent{Int,Int}(bad, 1)
        end
    end

    @testset "SpikeTrain construction and invariants" begin
        train = SpikeTrain(2, [0.3, 0.1, 0.2])
        @test train isa SpikeTrain{Float64,Int}
        @test neuron_id(train) == 2
        @test spike_times(train) == [0.1, 0.2, 0.3]
        @test issorted(spike_times(train))
        @test length(train) == 3
        @test !isempty(train)

        # caller's vector is neither mutated nor aliased by the copying constructor
        src = [0.3, 0.1, 0.2]
        copied = SpikeTrain(1, src)
        @test src == [0.3, 0.1, 0.2]
        @test spike_times(copied) !== src
        sorted_src = [0.1, 0.2]
        @test spike_times(SpikeTrain(1, sorted_src; sorted = true)) !== sorted_src

        # empty trains
        empty_train = SpikeTrain(5, Float64[])
        @test isempty(empty_train)
        @test length(empty_train) == 0
        @test spike_times(empty_train) == Float64[]
        @test SpikeTrain{Float64,Int}(5, Float64[]) == empty_train
        @test isempty(spike_events(empty_train))
        @test spike_count(spike_times(empty_train)) == 0

        # duplicate timestamps are preserved
        dup = SpikeTrain(1, [0.2, 0.1, 0.2, 0.2])
        @test spike_times(dup) == [0.1, 0.2, 0.2, 0.2]
        @test length(dup) == 4
        @test isi_stats(spike_times(dup)).min == 0.0

        # ordering: sorted=true and the parametric constructor reject unsorted input
        @test_throws ArgumentError SpikeTrain(1, [0.2, 0.1]; sorted = true)
        @test_throws ArgumentError SpikeTrain{Float64,Int}(1, [0.2, 0.1])
        @test SpikeTrain(1, [0.1, 0.1, 0.2]; sorted = true) ==
              SpikeTrain(1, [0.2, 0.1, 0.1])

        # sortedness follows isless so stored order matches SpikeEvent ordering
        @test_throws ArgumentError SpikeTrain(1, [0.0, -0.0]; sorted = true)
        @test_throws ArgumentError SpikeTrain{Float64,Int}(1, [0.0, -0.0])
        @test issorted(spike_times(SpikeTrain(1, [0.0, -0.0])))

        # non-finite times rejected on every path
        for bad in (NaN, Inf, -Inf)
            @test_throws ArgumentError SpikeTrain(1, [0.1, bad])
            @test_throws ArgumentError SpikeTrain(1, [0.1, bad]; sorted = true)
            @test_throws ArgumentError SpikeTrain{Float64,Int}(1, [0.1, bad])
        end
        # NaN raises ArgumentError (not InexactError) even when T cannot represent it
        for bad in (NaN, Inf, -Inf)
            @test_throws ArgumentError SpikeTrain{Int,Int}(1, [0.1, bad])
        end

        # numeric timestamp and id types
        @test SpikeTrain(1, Float32[0.2, 0.1]) isa SpikeTrain{Float32,Int}
        @test SpikeTrain(Int16(1), [3, 1, 2]) isa SpikeTrain{Int,Int16}
        @test spike_times(SpikeTrain(UInt32(1), [3, 1, 2])) == [1, 2, 3]
        @test spike_times(SpikeTrain(1, [1 // 2, 1 // 4])) == [1 // 4, 1 // 2]
        @test SpikeTrain{Float64,Int}(1, [1, 2]) isa SpikeTrain{Float64,Int}
        @test spike_times(SpikeTrain(1, view([0.3, 0.1, 0.2], 1:2))) == [0.1, 0.3]

        # equality and hashing
        @test SpikeTrain(1, [0.1, 0.2]) == SpikeTrain(1, [0.2, 0.1])
        @test SpikeTrain(1, [0.1]) != SpikeTrain(2, [0.1])
        @test hash(SpikeTrain(1, [0.1, 0.2])) == hash(SpikeTrain(1, [0.2, 0.1]))
        @test length(Set([SpikeTrain(1, [0.1]), SpikeTrain(1, [0.1])])) == 1
    end

    @testset "SpikeTrain works with timestamp-vector APIs" begin
        raw = [0.001, 0.005, 0.009, 0.040, 0.042, 0.044, 0.090]
        train = SpikeTrain(1, reverse(raw))
        st = spike_times(train)
        @test spike_count(st) == spike_count(raw)
        @test spike_density(st; t_start = 0.0, t_end = 0.1) ==
              spike_density(raw; t_start = 0.0, t_end = 0.1)
        @test isi_stats(st) == isi_stats(raw)
        @test detect_bursts(st; max_isi = 0.004) == detect_bursts(raw; max_isi = 0.004)
        @test windowed_spike_features(st; window_size = 0.03) ==
              windowed_spike_features(raw; window_size = 0.03)
        @test normalized_feature_vector(st) == normalized_feature_vector(raw)
        @test spike_count(spike_times(SpikeTrain(1, Float32[0.1, 0.2]))) == 2
        @test spike_count(spike_times(SpikeTrain(1, [1, 2, 3]))) == 3
    end

    @testset "train/event conversions with multiple neuron ids" begin
        a = SpikeTrain(2, [0.1, 0.3])
        b = SpikeTrain(1, [0.1, 0.2, 0.2])
        c = SpikeTrain(7, Float64[])

        ev_a = spike_events(a)
        @test ev_a == [SpikeEvent(0.1, 2), SpikeEvent(0.3, 2)]
        @test eltype(ev_a) == SpikeEvent{Float64,Int}

        merged = spike_events([a, b, c])
        @test eltype(merged) == SpikeEvent{Float64,Int}
        @test merged == [
            SpikeEvent(0.1, 1),
            SpikeEvent(0.1, 2),
            SpikeEvent(0.2, 1),
            SpikeEvent(0.2, 1),
            SpikeEvent(0.3, 2),
        ]
        @test issorted(merged)
        @test isempty(spike_events(SpikeTrain{Float64,Int}[]))

        trains = spike_trains(reverse(merged))
        @test trains == [b, a]
        @test map(neuron_id, trains) == [1, 2]
        @test all(tr -> issorted(spike_times(tr)), trains)
        @test isempty(spike_trains(SpikeEvent{Float64,Int}[]))
        @test spike_trains(spike_events([a, b])) == [b, a]

        # mixed numeric types are promoted
        mixed_trains = [SpikeTrain(Int16(1), Float32[0.5]), SpikeTrain(2, [0.25])]
        mixed_events = spike_events(mixed_trains)
        @test eltype(mixed_events) == SpikeEvent{Float64,Int}
        @test mixed_events == [SpikeEvent(0.25, 2), SpikeEvent(0.5, 1)]
        @test isempty(spike_events(SpikeTrain[]))

        mixed_in = SpikeEvent[SpikeEvent(Float32(0.5), Int16(1)), SpikeEvent(0.25, 2)]
        mixed_out = spike_trains(mixed_in)
        @test eltype(mixed_out) == SpikeTrain{Float64,Int}
        @test mixed_out == [SpikeTrain(1, [0.5]), SpikeTrain(2, [0.25])]
        @test isempty(spike_trains(SpikeEvent[]))
        @test eltype(spike_trains(SpikeEvent{Float32,Int16}[])) == SpikeTrain{Float32,Int16}
        @test eltype(spike_events(SpikeTrain{Int,UInt8}[])) == SpikeEvent{Int,UInt8}

        int_events = spike_events(SpikeTrain(UInt8(3), [5, 1]))
        @test int_events == [SpikeEvent(1, UInt8(3)), SpikeEvent(5, UInt8(3))]
        @test spike_trains(int_events) == [SpikeTrain(UInt8(3), [1, 5])]
    end

    @testset "conversion allocations" begin
        v = collect(0.0:0.001:1.0)
        alias_train(x) = SpikeTrain{Float64,Int}(1, x)
        copy_train(x) = SpikeTrain(1, x; sorted = true)
        times_of(tr) = spike_times(tr)
        alias_train(v)
        copy_train(v)
        times_of(alias_train(v))

        # sorted Vector{T} input is aliased: no copy, no allocation
        @test spike_times(alias_train(v)) === v
        @test (@allocated alias_train(v)) == 0
        tr = alias_train(v)
        @test (@allocated times_of(tr)) == 0

        # copying constructor allocates roughly one vector of length(v)
        @test (@allocated copy_train(v)) <= sizeof(v) + 256

        # one train -> events is a single output vector
        events_of(x) = spike_events(x)
        events_of(tr)
        @test (@allocated events_of(tr)) <=
              length(v) * sizeof(SpikeEvent{Float64,Int}) + 256
    end

    @testset "SpikeFeatureStream" begin
        @testset "empty and validation" begin
            s = SpikeFeatureStream(window = 0.1)
            @test snapshot(s) === nothing

            @test_throws ArgumentError SpikeFeatureStream(window = 0.0)
            @test_throws ArgumentError SpikeFeatureStream(window = -1.0)
            @test_throws ArgumentError SpikeFeatureStream(window = 0.1, step = 0.0)
            @test_throws ArgumentError SpikeFeatureStream(window = 0.1, t_start = NaN)
            @test_throws ArgumentError SpikeFeatureStream(window = 0.1, t_start = Inf)
            @test_throws ArgumentError SpikeFeatureStream(window = Inf)
            @test_throws ArgumentError SpikeFeatureStream(window = 1e300, step = 1.0)
            @test_throws ArgumentError SpikeFeatureStream(window = 0.1, max_isi = NaN)
        end

        @testset "invalid event times reject before mutating" begin
            s = SpikeFeatureStream(window = 0.1, step = 0.1, t_start = 0.0)
            push!(s, SpikeEvent(0.5, 1))
            # BigFloat time overflows Float64 -> Inf
            @test_throws ArgumentError push!(s, SpikeEvent(big(10.0)^400, 1))
            # finite but absurdly far from the anchor -> Int-range window index
            @test_throws ArgumentError push!(s, SpikeEvent(1e30, 1))
            # stream still usable afterwards (0.6 lands in window 6, alone)
            @test push!(s, SpikeEvent(0.6, 1)) == true
            @test snapshot(s).neurons[1].features.count == 1

            # many pre-anchor events stay evicted (bounded memory)
            s2 = SpikeFeatureStream(window = 0.1, step = 0.1, t_start = 10.0)
            for i = 1:1000
                push!(s2, SpikeEvent(0.001 * i, 1))
            end
            @test isempty(s2.times)
            @test snapshot(s2).neurons == NamedTuple[]
        end

        @testset "basic ingest and per-neuron snapshot" begin
            s = SpikeFeatureStream(window = 0.1, step = 0.01)
            @test push!(s, SpikeEvent(0.101, 17)) == true
            @test push!(s, SpikeEvent(0.104, 83)) == true
            @test push!(s, SpikeEvent(0.109, 17)) == true

            snap = snapshot(s)
            @test snap !== nothing
            @test snap.t_start == 0.101
            @test snap.t_end == 0.201
            @test snap.window == 0
            @test length(snap.neurons) == 2
            @test [row.neuron_id for row in snap.neurons] == [17, 83]

            row17 = snap.neurons[1]
            @test row17.features.count == 2
            @test row17.features.density == 20.0
            @test row17.features.isi_mean ≈ 0.008
            row83 = snap.neurons[2]
            @test row83.features.count == 1
            @test row83.features.density == 10.0
            @test row83.features.isi_mean == 0.0
        end

        @testset "late and out-of-order events" begin
            s = SpikeFeatureStream(window = 0.1, step = 0.1, t_start = 0.0)
            @test push!(s, SpikeEvent(0.2, 1)) == true
            @test push!(s, SpikeEvent(0.1, 1)) == false
            @test push!(s, SpikeEvent(0.15, 1)) == false
            @test s.dropped == 2
            @test push!(s, SpikeEvent(0.2, 2)) == true  # equal time is fine
            @test s.dropped == 2

            snap = snapshot(s)
            @test snap.window == 2
            @test snap.t_start == 0.2
            @test [row.neuron_id for row in snap.neurons] == [1, 2]
            @test snap.neurons[1].features.count == 1
        end

        @testset "tumbling windows match batch spike_features" begin
            rng = MersenneTwister(42)
            events = SpikeEvent[]
            for id in (1, 2, 3)
                t = 0.0
                while t < 2.0
                    t += 0.005 + rand(rng) * 0.045
                    push!(events, SpikeEvent(t, id))
                end
            end
            sort!(events)
            anchor = Float64(first(events).time)

            s = SpikeFeatureStream(window = 0.25, step = 0.25)
            by_window = Dict{Int,NamedTuple}()
            for ev in events
                push!(s, ev)
                snap = snapshot(s)
                snap === nothing || (by_window[snap.window] = snap)
            end
            @test length(by_window) >= 6

            for (k, snap) in by_window
                cur, nxt = snap.t_start, snap.t_end
                @test cur ≈ anchor + 0.25 * k
                @test nxt ≈ cur + 0.25
                for row in snap.neurons
                    id = row.neuron_id
                    ts = sort!(
                        Float64[
                            ev.time for
                            ev in events if ev.neuron_id == id && cur <= ev.time < nxt
                        ],
                    )
                    @test !isempty(ts)
                    ref = spike_features(ts; t_start = cur, t_end = nxt)
                    @test row.features == ref
                end
                # every accepted event in the window must appear under its unit
                expected_ids =
                    sort!(unique(ev.neuron_id for ev in events if cur <= ev.time < nxt))
                @test [row.neuron_id for row in snap.neurons] == expected_ids
            end
        end

        @testset "sliding windows: final snapshot matches batch" begin
            rng = MersenneTwister(7)
            events = SpikeEvent[]
            for id in (10, 20)
                t = 0.0
                while t < 1.0
                    t += 0.01 + rand(rng) * 0.07
                    push!(events, SpikeEvent(t, id))
                end
            end
            sort!(events)

            s = SpikeFeatureStream(window = 0.2, step = 0.05)
            for ev in events
                push!(s, ev)
            end
            snap = snapshot(s)
            @test snap !== nothing
            cur, nxt = snap.t_start, snap.t_end
            @test nxt - cur ≈ 0.2

            for row in snap.neurons
                id = row.neuron_id
                ts = sort!(
                    Float64[
                        ev.time for
                        ev in events if ev.neuron_id == id && cur <= ev.time < nxt
                    ],
                )
                ref = spike_features(ts; t_start = cur, t_end = nxt)
                @test row.features == ref
            end
        end

        @testset "window boundary ownership" begin
            # spike exactly on a step boundary belongs to the later window
            s = SpikeFeatureStream(window = 0.1, step = 0.1, t_start = 0.0)
            push!(s, SpikeEvent(0.1, 1))
            snap = snapshot(s)
            @test snap.window == 1
            @test snap.t_start == 0.1
            @test snap.neurons[1].features.count == 1

            # sliding window: event just below the end stays in the earlier
            # window only when no later step cell owns it
            s2 = SpikeFeatureStream(window = 0.15, step = 0.1, t_start = 0.0)
            push!(s2, SpikeEvent(0.14, 1))
            snap2 = snapshot(s2)
            @test snap2.window == 1
            @test snap2.t_start ≈ 0.1
            @test snap2.t_end ≈ 0.25
            @test snap2.neurons[1].features.count == 1
        end

        @testset "push! with SpikeTrain and custom burst params" begin
            s = SpikeFeatureStream(window = 0.1, step = 0.1; max_isi = 0.05, min_spikes = 2)
            train = SpikeTrain(5, [0.02, 0.03, 0.04, 0.11])
            @test push!(s, train) == 4
            snap = snapshot(s)
            @test snap.window == 0
            @test snap.neurons[1].neuron_id == 5
            @test snap.neurons[1].features.count == 4
            # ISIs: 0.01, 0.01, 0.07 -> one burst of 3 with max_isi=0.05, min_spikes=2
            @test snap.neurons[1].features.burst_count == 1

            # a late head is dropped while the newer tail is accepted
            train2 = SpikeTrain(5, [0.05, 0.5])
            @test push!(s, train2) == 1
            @test s.dropped == 1
            snap2 = snapshot(s)
            @test snap2.window == 4  # (0.5 - 0.02) / 0.1 -> 4
            @test snap2.neurons[1].features.count == 1
        end

        @testset "bounded memory on long streams" begin
            rng = MersenneTwister(123)
            s = SpikeFeatureStream(window = 0.05, step = 0.05)
            n_events = 0
            for i = 1:20000
                id = rand(rng, 1:50)
                t = 0.001 * i + rand(rng) * 0.0005
                # keep global order by construction: t increases with i
                push!(s, SpikeEvent(t, id))
                n_events += 1
            end
            buffered = sum(length, values(s.times))
            @test buffered <= 50 * 4  # ~50 units x events inside one 0.05 window horizon
            @test n_events == 20000
            @test length(s.times) <= 50
            snap = snapshot(s)
            @test snap !== nothing
        end

        @testset "determinism and t_start anchoring" begin
            events = [
                SpikeEvent(0.35, 1),
                SpikeEvent(0.36, 2),
                SpikeEvent(0.37, 1),
                SpikeEvent(0.55, 2),
                SpikeEvent(0.56, 1),
            ]
            s1 = SpikeFeatureStream(window = 0.1, step = 0.1; t_start = 0.0)
            s2 = SpikeFeatureStream(window = 0.1, step = 0.1; t_start = 0.0)
            for ev in events
                push!(s1, ev)
                push!(s2, ev)
            end
            @test snapshot(s1) == snapshot(s2)
            snap = snapshot(s1)
            @test snap.window == 5  # 0.56 -> floor((0.56-0.0)/0.1)
            @test snap.t_start ≈ 0.5

            # implicit anchor: first event sets t0
            s3 = SpikeFeatureStream(window = 0.1, step = 0.1)
            for ev in events
                push!(s3, ev)
            end
            snap3 = snapshot(s3)
            @test snap3.t_start ≈ 0.55
            @test snap3.window == 2  # (0.56-0.35)/0.1 -> 2

            # events before t_start are accepted but outside every window
            s4 = SpikeFeatureStream(window = 0.1, step = 0.1; t_start = 1.0)
            push!(s4, SpikeEvent(0.5, 1))
            snap4 = snapshot(s4)
            @test snap4.window == 0
            @test snap4.neurons == NamedTuple[]
        end

        @testset "gap windows discard unusable events" begin
            s = SpikeFeatureStream(window = 0.05, step = 0.2, t_start = 0.0)
            push!(s, SpikeEvent(0.01, 1))
            push!(s, SpikeEvent(0.08, 1))  # gap: window0 ends 0.05, next starts 0.2
            push!(s, SpikeEvent(0.12, 1))  # gap
            @test s.times[1] == [0.01]
            snap = snapshot(s)
            @test snap.window == 0
            @test snap.neurons[1].features.count == 1
            push!(s, SpikeEvent(0.21, 1))
            @test s.times[1] == [0.21]
            @test snapshot(s).neurons[1].features.count == 1
        end

        @testset "non-Int neuron ids and Float32 times" begin
            s = SpikeFeatureStream{Int32}(window = 0.1, step = 0.1)
            push!(s, SpikeEvent(0.05f0, Int32(7)))
            push!(s, SpikeEvent(0.06f0, Int32(3)))
            snap = snapshot(s)
            @test [row.neuron_id for row in snap.neurons] == Int32[3, 7]
            @test snap.neurons[2].features.count == 1
        end
    end

    @testset "population features" begin
        pop = [
            SpikeTrain(1, [0.1, 0.2, 0.3]),
            SpikeTrain(7, [0.15]),
            SpikeTrain(1003, Float64[]),
        ]

        @testset "batch snapshot over SpikeTrain vectors" begin
            snap = population_spike_features(pop; t_start = 0.0, t_end = 1.0)
            @test snap isa PopulationFeatureSnapshot
            @test snap.duration == 1.0
            @test snap.n_total_units == 3
            @test snap.n_observed_units == 3
            @test snap.n_active_units == 2
            @test snap.total_spike_count == 4
            @test snap.population_rate == 4.0
            @test snap.mean_unit_rate ≈ 4 / 3
            @test snap.active_fraction ≈ 2 / 3
            @test snap.count_mean ≈ 4 / 3
            @test snap.count_variance ≈ (25 + 1 + 16) / 27
            @test snap.fano_factor ≈ snap.count_variance / snap.count_mean

            # repeated calls are pure: same input, same snapshot, no mutation
            @test population_spike_features(pop; t_start = 0.0, t_end = 1.0) == snap
            @test spike_times(pop[1]) == [0.1, 0.2, 0.3]

            # inferred duration spans first/last spike of the whole population
            inferred = population_spike_features(pop)
            @test inferred.duration ≈ 0.2
            @test inferred.population_rate ≈ 4 / 0.2

            # a single SpikeTrain is a one-unit population
            solo = population_spike_features(SpikeTrain(4, [0.1, 0.9]))
            @test solo.n_total_units == 1
            @test solo.n_active_units == 1
            @test solo.population_rate ≈ 2 / 0.8
            @test solo.count_variance == 0.0
            @test solo.fano_factor == 0.0
        end

        @testset "batch snapshot over SpikeEvent vectors" begin
            events = [
                SpikeEvent(0.05, 1),
                SpikeEvent(0.5, 1),
                SpikeEvent(0.9, 1),
                SpikeEvent(0.5, 2),
                SpikeEvent(0.9, 3),
            ]
            snap = population_spike_features(events; t_start = 0.4, t_end = 0.6)
            @test snap.duration ≈ 0.2
            # events outside the window still mark their unit as observed
            @test snap.n_observed_units == 3
            @test snap.n_active_units == 2
            @test snap.total_spike_count == 2
            @test snap.population_rate ≈ 10.0
            @test snap.active_fraction ≈ 2 / 3

            # expanding trains to events drops declared-silent units: the empty
            # train above is unobserved, so the denominator shrinks accordingly
            ev_pop = spike_events(pop)
            ev_snap = population_spike_features(ev_pop; t_start = 0.0, t_end = 1.0)
            @test ev_snap.n_observed_units == 2
            @test ev_snap.n_active_units == 2
            @test ev_snap.mean_unit_rate == 2.0
            @test ev_snap.active_fraction == 1.0
            @test ev_snap.count_mean == 2.0
            @test ev_snap.count_variance == 1.0
            @test ev_snap.fano_factor == 0.5
        end

        @testset "n_total_units denominator semantics" begin
            snap = population_spike_features(
                pop;
                t_start = 0.0,
                t_end = 1.0,
                n_total_units = 5,
            )
            @test snap.n_total_units == 5
            @test snap.n_observed_units == 3
            @test snap.mean_unit_rate ≈ 4 / 5
            @test snap.active_fraction ≈ 2 / 5
            @test snap.count_mean ≈ 4 / 5
            # absent declared units contribute (0 - mean)^2 each
            @test snap.count_variance ≈ (4.84 + 0.04 + 0.64 + 2 * 0.64) / 5
            @test snap.fano_factor ≈ snap.count_variance / snap.count_mean

            @test_throws ArgumentError population_spike_features(pop; n_total_units = 2)
            @test_throws ArgumentError population_spike_features(pop; n_total_units = -1)

            # declared-but-absent units on an empty population stay zeros
            ghost = population_spike_features(
                SpikeTrain{Float64,Int}[];
                t_start = 0.0,
                t_end = 1.0,
                n_total_units = 4,
            )
            @test ghost.n_total_units == 4
            @test ghost.n_observed_units == 0
            @test ghost.count_mean == 0.0
            @test isnan(ghost.fano_factor)
        end

        @testset "empty, silent, and single-unit populations" begin
            for empty_pop in (SpikeTrain{Float64,Int}[], SpikeEvent{Float64,Int}[])
                snap = population_spike_features(empty_pop; t_start = 0.0, t_end = 1.0)
                @test snap.duration == 1.0
                @test snap.n_total_units == 0
                @test snap.n_observed_units == 0
                @test snap.n_active_units == 0
                @test snap.total_spike_count == 0
                @test snap.population_rate == 0.0
                @test snap.active_fraction == 0.0
                @test snap.count_mean == 0.0
                @test snap.count_variance == 0.0
                @test isnan(snap.fano_factor)
                @test isempty(unit_firing_rates(empty_pop))
            end

            # no edges at all: inferred duration is 0 and rates stay 0
            bare = population_spike_features(SpikeTrain[])
            @test bare.duration == 0.0
            @test isnan(bare.fano_factor)

            # silent population: all observed units carry zero in-window spikes
            silent = population_spike_features(pop; t_start = 2.0, t_end = 3.0)
            @test silent.duration == 1.0
            @test silent.n_observed_units == 3
            @test silent.n_active_units == 0
            @test silent.population_rate == 0.0
            @test silent.count_mean == 0.0
            @test isnan(silent.fano_factor)

            # inverted window: duration clamps to 0, nothing is counted
            inverted = population_spike_features(pop; t_start = 1.0, t_end = 0.0)
            @test inverted.duration == 0.0
            @test inverted.total_spike_count == 0
            @test inverted.population_rate == 0.0
            @test isnan(inverted.fano_factor)

            # zero-length window on a spike instant: counts are reported while
            # rate fields stay 0 and count statistics remain meaningful
            instant = population_spike_features(pop; t_start = 0.2, t_end = 0.2)
            @test instant.duration == 0.0
            @test instant.total_spike_count == 1
            @test instant.n_active_units == 1
            @test instant.population_rate == 0.0
            @test instant.mean_unit_rate == 0.0
            @test instant.count_mean ≈ 1 / 3
            @test instant.count_variance ≈ 2 / 9
            @test instant.fano_factor ≈ 2 / 3
        end

        @testset "heterogeneous and homogeneous rates" begin
            het = [
                SpikeTrain(1, [0.1, 0.2, 0.3, 0.4, 0.5]),
                SpikeTrain(2, [0.1]),
                SpikeTrain(3, Float64[]),
            ]
            snap = population_spike_features(het; t_start = 0.0, t_end = 2.0)
            @test snap.count_mean ≈ 2.0
            @test snap.count_variance ≈ (9 + 1 + 4) / 3
            @test snap.fano_factor ≈ snap.count_variance / 2

            # identical per-unit counts: zero count variance, Fano factor 0
            hom = [SpikeTrain(id, [0.1, 0.2, 0.3]) for id = 1:3]
            hom_snap = population_spike_features(hom; t_start = 0.0, t_end = 1.0)
            @test hom_snap.count_variance == 0.0
            @test hom_snap.fano_factor == 0.0
        end

        @testset "sparse and non-contiguous unit ids" begin
            sparse_pop = [
                SpikeTrain(-3, [0.1]),
                SpikeTrain(0, Float64[]),
                SpikeTrain(Int16(7), [0.2, 0.4]),
                SpikeTrain(10^6, [0.3]),
            ]
            snap = population_spike_features(sparse_pop; t_start = 0.0, t_end = 1.0)
            @test snap.n_observed_units == 4
            @test snap.n_active_units == 3

            rates = unit_firing_rates(sparse_pop; t_start = 0.0, t_end = 1.0)
            @test [r.neuron_id for r in rates] == [-3, 0, 7, 10^6]
            @test rates[1].neuron_id isa Int
            @test rates[3].neuron_id isa Int16
            @test rates[2].spike_count == 0
            @test rates[2].firing_rate == 0.0
            @test rates[3].firing_rate == 2.0

            # out-of-window spikes still produce a zero-rate row
            late = unit_firing_rates(sparse_pop; t_start = 0.0, t_end = 0.05)
            @test length(late) == 4
            @test all(r -> r.spike_count == 0 && r.firing_rate == 0.0, late)
        end

        @testset "per-unit firing rates" begin
            rates = unit_firing_rates(pop; t_start = 0.0, t_end = 1.0)
            @test length(rates) == 3
            @test [r.neuron_id for r in rates] == [1, 7, 1003]
            @test [r.spike_count for r in rates] == [3, 1, 0]
            @test [r.firing_rate for r in rates] == [3.0, 1.0, 0.0]

            solo = unit_firing_rates(SpikeTrain(9, [0.1, 0.9]))
            @test length(solo) == 1
            @test solo[1].neuron_id == 9
            @test solo[1].firing_rate ≈ 2 / 0.8

            # sum of per-unit rates is the population rate
            snap = population_spike_features(pop; t_start = 0.0, t_end = 1.0)
            @test sum(r.firing_rate for r in rates) == snap.population_rate
        end

        @testset "inferred duration needs two spikes" begin
            # one spike + one explicit bound: inferred edge is undefined, so
            # duration (and therefore every rate) is 0 — same rule as
            # spike_density — while the count is still reported
            for singleton in ([SpikeTrain(1, [0.999])], [SpikeEvent(0.999, 1)])
                snap = population_spike_features(singleton; t_end = 1.0)
                @test snap.duration == 0.0
                @test snap.total_spike_count == 1
                @test snap.population_rate == 0.0
                rates = unit_firing_rates(singleton; t_end = 1.0)
                @test rates[1].spike_count == 1
                @test rates[1].firing_rate == 0.0
            end
            for singleton in ([SpikeTrain(1, [0.5])], [SpikeEvent(0.5, 1)])
                snap = population_spike_features(singleton; t_start = 0.0)
                @test snap.duration == 0.0
                @test snap.total_spike_count == 1
                @test snap.population_rate == 0.0
                @test unit_firing_rates(singleton; t_start = 0.0)[1].firing_rate == 0.0
            end

            # both bounds explicit: a single spike keeps a positive duration
            snap = population_spike_features(
                [SpikeTrain(1, [0.5])];
                t_start = 0.0,
                t_end = 1.0,
            )
            @test snap.duration == 1.0
            @test snap.population_rate == 1.0
            @test unit_firing_rates([SpikeEvent(0.5, 1)]; t_start = 0.0, t_end = 1.0)[1].firing_rate ==
                  1.0

            # two or more spikes still infer a positive span
            two = population_spike_features(
                [SpikeTrain(1, [0.5]), SpikeTrain(2, [0.9])];
                t_start = 0.0,
            )
            @test two.duration ≈ 0.9
            @test two.population_rate ≈ 2 / 0.9
        end

        @testset "large integer timestamps keep their precision" begin
            # Float64 spacing around b is 256: distinct integer endpoints and
            # spike times collapse to one value when converted first
            b = 1_700_000_000_000_000_000
            @test Float64(b) == Float64(b + 100)

            for p in ([SpikeTrain(1, [b + 50])], [SpikeEvent(b + 50, 1)])
                snap = population_spike_features(p; t_start = b, t_end = b + 100)
                @test snap.duration == 100.0
                @test snap.total_spike_count == 1
                @test snap.population_rate ≈ 0.01
                rates = unit_firing_rates(p; t_start = b, t_end = b + 100)
                @test rates[1].spike_count == 1
                @test rates[1].firing_rate ≈ 0.01
            end

            # an event just outside the window must stay uncounted
            for p in ([SpikeTrain(1, [b + 150])], [SpikeEvent(b + 150, 1)])
                snap = population_spike_features(p; t_start = b, t_end = b + 100)
                @test snap.total_spike_count == 0
                @test snap.n_active_units == 0
                @test snap.population_rate == 0.0
            end

            # inferred edges keep integer precision too
            snap = population_spike_features([
                SpikeTrain(1, [b + 50]),
                SpikeTrain(2, [b + 60]),
            ])
            @test snap.duration == 10.0
            @test snap.population_rate ≈ 0.2
        end

        @testset "signed-zero bounds are representation-independent" begin
            neg_train = SpikeTrain(7, [-0.0])
            pos_train = SpikeTrain(7, [0.0])
            for (tr, t_start, t_end) in (
                (neg_train, 0.0, 1.0),
                (pos_train, -1.0, -0.0),
                (neg_train, -0.0, -0.0),
                (pos_train, -0.0, -0.0),
                (neg_train, -0.0, 1.0),
            )
                by_train = population_spike_features([tr]; t_start = t_start, t_end = t_end)
                by_event = population_spike_features(
                    spike_events(tr);
                    t_start = t_start,
                    t_end = t_end,
                )
                @test by_train.total_spike_count == by_event.total_spike_count
                @test by_train.population_rate == by_event.population_rate
            end

            # isless orders -0.0 before 0.0, so a -0.0 spike sits below a 0.0
            # lower bound while a 0.0 spike sits above a -0.0 upper bound
            @test population_spike_features([neg_train]; t_start = 0.0, t_end = 1.0).total_spike_count ==
                  0
            @test population_spike_features(
                spike_events(neg_train);
                t_start = 0.0,
                t_end = 1.0,
            ).total_spike_count == 0
            @test population_spike_features([pos_train]; t_start = -1.0, t_end = -0.0).total_spike_count ==
                  0
            @test population_spike_features(
                spike_events(neg_train);
                t_start = -0.0,
                t_end = -0.0,
            ).total_spike_count == 1
        end

        @testset "mixed timestamp types keep original extrema" begin
            # `min`/`max` promote a signed and unsigned time to UInt and
            # throw; `isless` keeps each original timestamp
            mixed_sign = SpikeEvent[SpikeEvent(-1, 1), SpikeEvent(UInt(1), 2)]
            snap = population_spike_features(mixed_sign; t_start = -2, t_end = 2)
            @test snap.total_spike_count == 2
            @test snap.duration == 4.0
            @test snap.population_rate == 0.5
            inferred = population_spike_features(mixed_sign)
            @test inferred.duration == 2.0
            @test inferred.population_rate == 1.0

            # Float32(2^24) promotes Int times back down to Float32, which
            # would erase the one-unit span; extrema keep the raw values
            float_int =
                SpikeEvent[SpikeEvent(Float32(16_777_216), 1), SpikeEvent(16_777_217, 2)]
            snap = population_spike_features(float_int)
            @test snap.duration == 1.0
            @test snap.total_spike_count == 2
            @test snap.population_rate == 2.0

            # same invariants through the SpikeTrain path
            sign_trains = SpikeTrain[SpikeTrain(1, Int[-1]), SpikeTrain(2, UInt64[0x1])]
            @test population_spike_features(sign_trains; t_start = -2, t_end = 2).total_spike_count ==
                  2
            @test population_spike_features(sign_trains).duration == 2.0
            f32_trains =
                SpikeTrain[SpikeTrain(1, Float32[16_777_216]), SpikeTrain(2, [16_777_217])]
            @test population_spike_features(f32_trains).duration == 1.0
        end

        @testset "float bounds share the spike_count window domain" begin
            train = SpikeTrain(1, [1 // 3])
            # `spike_count` converts both sides to Float64, so the rational
            # spike lands exactly on the Float64 upper bound
            @test spike_count(spike_times(train); t_start = 0.0, t_end = 1 / 3) == 1
            for p in ([train], [SpikeEvent(1 // 3, 1)])
                snap = population_spike_features(p; t_start = 0.0, t_end = 1 / 3)
                @test snap.total_spike_count == 1
                @test unit_firing_rates(p; t_start = 0.0, t_end = 1 / 3)[1].spike_count == 1
            end
            # the same edge on the lower bound
            @test population_spike_features([train]; t_start = 1 / 3, t_end = 1.0).total_spike_count ==
                  1

            # exact bounds still compare the stored values: the Float64
            # instant below the rational bound stays in, a rational instant
            # strictly above it stays out
            @test population_spike_features(
                [SpikeEvent(1 / 3, 1)];
                t_start = 0,
                t_end = 1 // 3,
            ).total_spike_count == 1
            @test population_spike_features(
                [SpikeEvent(1 // 3 + big(1) // big(10)^30, 1)];
                t_start = 0,
                t_end = 1 // 3,
            ).total_spike_count == 0
        end

        @testset "train and event overloads count identical windows" begin
            # Float64() rounds numerator and denominator separately, so it
            # can reverse the order of close rationals: these times are
            # exactly increasing but their Float64 images swap
            n = Int64(2)^53
            times = Rational{Int64}[(n+6)//(n+5), (n+4)//(n+3)]
            @test issorted(times)
            @test Float64(times[1]) > Float64(times[2])
            trs = SpikeTrain[SpikeTrain(1, times)]
            evs = SpikeEvent[SpikeEvent(t, 1) for t in times]
            for p in (trs, evs)
                snap = population_spike_features(p; t_start = 0.0, t_end = 1.0)
                @test snap.total_spike_count == 1
            end

            # one-sided exact windows search the sorted rational times
            # directly: the omitted bound's ±Inf sentinel must not force a
            # misordered Float64-image scan
            for p in (trs, evs)
                @test population_spike_features(p; t_end = times[1]).total_spike_count == 1
                @test population_spike_features(p; t_start = times[2]).total_spike_count ==
                      1
                @test population_spike_features(p; t_end = times[2]).total_spike_count == 2
            end

            # Float64(b-1) rounds up to Float64(b): the exact window is
            # inverted even though the spike meets each converted bound
            b = 1_700_000_000_000_000_000
            @test Float64(b - 1) == Float64(b)
            for p in (SpikeEvent[SpikeEvent(b - 1, 1)], SpikeTrain[SpikeTrain(1, [b - 1])])
                snap = population_spike_features(p; t_start = Float64(b), t_end = b - 1)
                @test snap.total_spike_count == 0
                # the unit is still observed even with nothing counted
                @test snap.n_observed_units == 1
            end
        end

        @testset "high-precision bounds keep duration precision" begin
            # BigFloat bounds closer than Float64 resolution: the window
            # still contains a spike, so the rate must not be 0
            b1 = big"1.0"
            b2 = b1 + big"1e-70"
            mid = b1 + big"5e-71"
            for p in ([SpikeEvent(mid, 1)], [SpikeTrain(1, [mid])])
                snap = population_spike_features(p; t_start = b1, t_end = b2)
                @test snap.total_spike_count == 1
                @test snap.duration == Float64(b2 - b1)
                @test snap.duration > 0
                @test snap.population_rate == 1 / snap.duration
            end

            # rational bounds below Float64 resolution keep an exact span
            r1 = 1 // 3
            r2 = r1 + big(1) // big(10)^40
            snap = population_spike_features([SpikeEvent(r1, 1)]; t_start = r1, t_end = r2)
            @test snap.total_spike_count == 1
            @test snap.duration == 1e-40
            @test snap.population_rate == 1 / snap.duration

            # rational endpoints over huge integers keep their span too
            b = 1_700_000_000_000_000_000
            @test Float64(b // 1) == Float64((b + 100) // 1)
            snap = population_spike_features(
                [SpikeTrain(1, [b + 50])];
                t_start = b // 1,
                t_end = (b + 100) // 1,
            )
            @test snap.duration == 100.0
            @test snap.population_rate ≈ 0.01
        end

        @testset "infinite bounds never make duration NaN" begin
            # equal infinite bounds span a zero-length window — Inf - Inf
            # would otherwise produce NaN, violating duration >= 0
            for p in ([SpikeEvent(0.5, 1)], [SpikeTrain(1, [0.5])])
                snap = population_spike_features(p; t_start = Inf, t_end = Inf)
                @test snap.duration == 0.0
                @test snap.total_spike_count == 0
                snap = population_spike_features(p; t_start = -Inf, t_end = -Inf)
                @test snap.duration == 0.0
                @test snap.total_spike_count == 0
                snap = population_spike_features(p; t_start = Inf, t_end = 5.0)
                @test snap.duration == 0.0
                snap = population_spike_features(p; t_start = -Inf, t_end = Inf)
                @test snap.duration == Inf
            end
        end

        @testset "duration ignores ambient BigFloat precision" begin
            # at 8-bit ambient precision BigFloat() rounds both Float64
            # endpoints to 1.0, so a BigFloat subtraction would report a
            # zero duration for a window that still contains the spike
            lo, hi = 1.0, 1.0 + 2.0^-10
            ev = SpikeEvent(lo + 2.0^-11, 1)
            for (a, b) in ((lo, hi), (1, hi), (lo, 1 + 1 // 1024), (1 // 1, hi))
                snap = setprecision(BigFloat, 8) do
                    population_spike_features([ev]; t_start = a, t_end = b)
                end
                @test snap.duration == 2.0^-10
                @test snap.population_rate ≈ 2.0^10
                @test snap.total_spike_count == 1
            end
        end

        @testset "omitted bounds never convert or scan timestamps" begin
            # an omitted bound's ±Inf sentinel is internal — an exact-bound
            # window over timestamps without a Float64 conversion must not
            # reach for Float64 at all
            n = 4096
            ticks = CmpTick.(1:n)
            trains = SpikeTrain[SpikeTrain(1, ticks)]
            events = SpikeEvent[SpikeEvent(t, 1) for t in ticks]
            for p in (events, trains)
                @test population_spike_features(p; t_end = 2).total_spike_count == 2
                @test population_spike_features(p; t_start = n - 1).total_spike_count == 2
                @test only(unit_firing_rates(p; t_end = 2)).spike_count == 2
            end
            @test population_spike_features([SpikeEvent(CmpTick(1), 1)]; t_end = 2).total_spike_count ==
                  1
            @test population_spike_features([SpikeTrain(1, [CmpTick(1)])]; t_end = 2).duration ==
                  0.0

            # sorted trains take the binary-search path: ~log2(n) bound
            # comparisons, not one per spike — a full predicate scan fails
            TICK_BOUND_CMPS[] = 0
            population_spike_features(trains; t_end = 2)
            @test 0 < TICK_BOUND_CMPS[] <= 64
            TICK_BOUND_CMPS[] = 0
            population_spike_features(trains; t_start = n - 1)
            @test 0 < TICK_BOUND_CMPS[] <= 64

            # an explicitly float bound still uses the Float64-image domain
            @test_throws MethodError population_spike_features(events; t_end = 2.0)
        end

        @testset "SpikeFeatureStream integration" begin
            stream = SpikeFeatureStream(window = 0.1, step = 0.1, t_start = 0.0)
            @test population_spike_features(stream) === nothing
            @test unit_firing_rates(stream) === nothing

            for ev in (SpikeEvent(0.21, 1), SpikeEvent(0.22, 2), SpikeEvent(0.23, 1))
                push!(stream, ev)
            end
            snap = population_spike_features(stream)
            @test snap isa PopulationFeatureSnapshot
            @test snap.duration ≈ 0.1
            @test snap.n_observed_units == 2
            @test snap.n_active_units == 2
            @test snap.total_spike_count == 3
            @test snap.population_rate ≈ 30.0
            @test snap.mean_unit_rate ≈ 15.0
            @test snap.count_mean ≈ 1.5
            @test snap.count_variance ≈ 0.25
            @test snap.fano_factor ≈ 1 / 6

            rates = unit_firing_rates(stream)
            @test [r.neuron_id for r in rates] == [1, 2]
            @test issorted([r.neuron_id for r in rates])
            @test [r.spike_count for r in rates] == [2, 1]
            @test rates[1].firing_rate ≈ 20.0
            @test rates[2].firing_rate ≈ 10.0

            # stream snapshot rows equal the equivalent batch window
            batch = population_spike_features(
                [SpikeEvent(0.21, 1), SpikeEvent(0.22, 2), SpikeEvent(0.23, 1)];
                t_start = 0.2,
                t_end = 0.3,
            )
            @test batch.total_spike_count == snap.total_spike_count
            @test batch.count_mean ≈ snap.count_mean
            @test batch.count_variance ≈ snap.count_variance

            # n_total_units applies on the stream path too
            wide = population_spike_features(stream; n_total_units = 10)
            @test wide.n_total_units == 10
            @test wide.active_fraction ≈ 2 / 10
        end
    end

    @testset "frozen spike fixtures (LIM-41)" begin
        fixture_path = joinpath(@__DIR__, "fixtures", "spike_vectors.json")
        @test isfile(fixture_path)

        fixture = JSON.parsefile(fixture_path)
        @test haskey(fixture, "tolerance")
        @test haskey(fixture, "cases")
        cases = fixture["cases"]
        @test length(cases) >= 2

        tol = Float64(fixture["tolerance"])
        required_expect_keys = (
            "spike_count",
            "spike_density",
            "isi_stats",
            "detect_bursts",
            "windowed_spike_features",
            "normalized_feature_vector",
        )

        for case in cases
            cname = get(case, "name", "<unnamed>")
            @testset "$cname" begin
                @test haskey(case, "name")
                @test haskey(case, "spike_times")
                @test haskey(case, "expect")
                expect = case["expect"]
                for key in required_expect_keys
                    @test haskey(expect, key)
                end

                times = Float64.(case["spike_times"])

                # --- spike_count ---
                got_count = spike_count(times)
                @test got_count == Int(expect["spike_count"])
                @test got_count >= 0

                # --- spike_density (full window from fixture) ---
                dens_exp = expect["spike_density"]
                dens_start = Float64(dens_exp["t_start"])
                dens_end = Float64(dens_exp["t_end"])
                got_density = spike_density(times; t_start = dens_start, t_end = dens_end)
                @test isapprox(
                    got_density,
                    Float64(dens_exp["value"]);
                    atol = tol,
                    rtol = 0,
                )
                @test got_density >= 0

                # windowed density over the same bounds (count / duration)
                windowed_count = spike_count(times; t_start = dens_start, t_end = dens_end)
                duration = dens_end - dens_start
                if length(times) >= 2 && duration > 0
                    @test isapprox(
                        got_density,
                        windowed_count / duration;
                        atol = tol,
                        rtol = 0,
                    )
                end

                # --- isi_stats ---
                stats = isi_stats(times)
                isi_exp = expect["isi_stats"]
                for field in ("mean", "std", "min", "max", "cv")
                    got = getfield(stats, Symbol(field))
                    @test isapprox(got, Float64(isi_exp[field]); atol = tol, rtol = 0)
                    @test got >= 0
                end

                # --- detect_bursts ---
                burst_exp = expect["detect_bursts"]
                max_isi = Float64(burst_exp["max_isi"])
                min_spikes = Int(burst_exp["min_spikes"])
                bursts = detect_bursts(times; max_isi = max_isi, min_spikes = min_spikes)
                ranges_exp = burst_exp["ranges"]
                @test length(bursts) == length(ranges_exp)
                for (burst, range_pair) in zip(bursts, ranges_exp)
                    @test first(burst) == Int(range_pair[1])
                    @test last(burst) == Int(range_pair[2])
                    @test burst isa UnitRange{Int}
                end

                # --- windowed_spike_features ---
                win_exp = expect["windowed_spike_features"]
                feats = windowed_spike_features(
                    times;
                    window_size = Float64(win_exp["window_size"]),
                    step = Float64(win_exp["step"]),
                    t_start = Float64(win_exp["t_start"]),
                    t_end = Float64(win_exp["t_end"]),
                )
                windows_exp = win_exp["windows"]
                @test length(feats) == length(windows_exp)
                for (feat, wexp) in zip(feats, windows_exp)
                    @test isapprox(
                        feat.t_start,
                        Float64(wexp["t_start"]);
                        atol = tol,
                        rtol = 0,
                    )
                    @test isapprox(feat.t_end, Float64(wexp["t_end"]); atol = tol, rtol = 0)
                    @test feat.count == Int(wexp["count"])
                    @test isapprox(
                        feat.density,
                        Float64(wexp["density"]);
                        atol = tol,
                        rtol = 0,
                    )
                    @test isapprox(
                        feat.isi_mean,
                        Float64(wexp["isi_mean"]);
                        atol = tol,
                        rtol = 0,
                    )
                    @test isapprox(
                        feat.isi_cv,
                        Float64(wexp["isi_cv"]);
                        atol = tol,
                        rtol = 0,
                    )
                    @test feat.burst_count == Int(wexp["burst_count"])

                    # documented range invariants
                    @test feat.count >= 0
                    @test feat.density >= 0
                    @test feat.isi_mean >= 0
                    @test feat.isi_cv >= 0
                    @test feat.burst_count >= 0
                end

                # --- normalized_feature_vector ---
                nfv_exp = expect["normalized_feature_vector"]
                nfv = normalized_feature_vector(
                    times;
                    t_start = Float64(nfv_exp["t_start"]),
                    t_end = Float64(nfv_exp["t_end"]),
                    max_density = Float64(nfv_exp["max_density"]),
                )
                expected_vec = Float64.(nfv_exp["value"])
                @test length(nfv) == 4
                @test length(expected_vec) == 4
                @test all(isapprox.(nfv, expected_vec; atol = tol, rtol = 0))
                @test all(0.0 .<= nfv .<= 1.0)
            end
        end
    end

end
