# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using SpikeStream

const CONVERSION_COUNT = Ref(0)

struct ConversionProbe <: Real
    value::Float64
end
Base.Float64(x::ConversionProbe) = (CONVERSION_COUNT[] += 1; x.value)
Base.isfinite(x::ConversionProbe) = isfinite(x.value)

@testset "SpikeStream" begin

    @testset "Package loads" begin
        @test @isdefined(SpikeStream)
        @test SpikeStream isa Module
        @test isdefined(SpikeStream, :spike_count)
        @test isdefined(SpikeStream, :spike_density)
        @test isdefined(SpikeStream, :isi_stats)
        @test isdefined(SpikeStream, :detect_bursts)
        @test isdefined(SpikeStream, :windowed_spike_features)
        @test isdefined(SpikeStream, :normalized_feature_vector)
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
