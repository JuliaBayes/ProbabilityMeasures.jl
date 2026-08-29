using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Xoshiro
using Test

reference(d) = d.μ + d.σ * Distributions.TDist(d.ν)

reference_logpdf(d, x) = Distributions.logpdf(reference(d), x)

@testset "conformance" begin
    ds = (TDist(5.0, 0.0, 1.0), TDist(8.0, -2.5, 0.5), TDist(6.0f0, 3.0f0, 2.0f0))
    for d in ds
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "reference numerics against Distributions.jl" begin
    for d in (TDist(5.0, 0.0, 1.0), TDist(8.0, -2.5, 0.5), TDist(2.5, 1.0, 3.0))
        r = reference(d)
        for x in (-40.0, -2.0, d.μ, 1.5, 30.0)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
            @test cdf(d, x) ≈ Distributions.cdf(r, x)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
        end
        for p in (1e-10, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-10)
            @test quantile(d, p) ≈ Distributions.quantile(r, p)
        end
        @test entropy(d) ≈ Distributions.entropy(r)
        @test var(d) ≈ Distributions.var(r)
    end
end

@testset "one degree of freedom is Cauchy" begin
    d, c = TDist(1.0, -2.0, 3.0), Cauchy(-2.0, 3.0)
    for x in (-100.0, -2.0, 0.5, 20.0)
        @test logdensityof(d, x) ≈ logdensityof(c, x)
        @test cdf(d, x) ≈ cdf(c, x)
        @test ccdf(d, x) ≈ ccdf(c, x)
    end
    for p in (1e-8, 0.25, 0.5, 0.9)
        @test quantile(d, p) ≈ quantile(c, p)
    end
    @test entropy(d) ≈ entropy(c)
    @test median(d) == median(c)
end

@testset "many degrees of freedom approach the normal" begin
    d, n = TDist(1.0e7, 1.5, 2.0), Normal(1.5, 2.0)
    for x in (-4.0, 1.5, 6.0)
        @test logdensityof(d, x) ≈ logdensityof(n, x) atol = 1e-5
        @test cdf(d, x) ≈ cdf(n, x) atol = 1e-6
    end
    @test entropy(d) ≈ entropy(n) atol = 1e-6
    @test var(d) ≈ var(n) rtol = 1e-6
end

@testset "moments exist only below the degrees of freedom" begin
    @test mean(TDist(0.5, 2.0, 1.0)) |> isnan
    @test mean(TDist(1.0, 2.0, 1.0)) |> isnan
    @test mean(TDist(1.5, 2.0, 1.0)) == 2.0

    @test var(TDist(0.5, 0.0, 1.0)) |> isnan
    @test var(TDist(1.0, 0.0, 1.0)) |> isnan
    @test var(TDist(1.5, 0.0, 2.0)) == Inf
    @test var(TDist(2.0, 0.0, 2.0)) == Inf
    @test var(TDist(4.0, 0.0, 2.0)) == 4 * 4 / 2
    @test std(TDist(4.0, 0.0, 2.0)) == sqrt(8.0)

    # The median is the location whatever the degrees of freedom.
    @test median(TDist(0.5, 2.0, 1.0)) == 2.0
    @test quantile(TDist(0.5, 2.0, 1.0), 0.5) == 2.0
end

@testset "parameters and construction" begin
    @test typeof(TDist(5, 0.0f0, 1)) === TDist{Int,Float32,Int}
    @test TDist(4.0) === TDist(4.0, 0.0, 1.0)
    @test params(TDist(5.0, 1.0, 2.0)) === (ν=5.0, μ=1.0, σ=2.0)

    for bad in (
        TDist(0.0, 0.0, 1.0),
        TDist(-1.0, 0.0, 1.0),
        TDist(Inf, 0.0, 1.0),
        TDist(NaN, 0.0, 1.0),
        TDist(5.0, Inf, 1.0),
        TDist(5.0, 0.0, 0.0),
        TDist(5.0, 0.0, -1.0),
        TDist(5.0, 0.0, Inf),
    )
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, 0.5))
        @test isnan(quantile(bad, 0.25))
        @test isnan(cdf(bad, 0.5))
    end
    @test checkparams(TDist(5.0, 0.0, 1.0))
end

@testset "stable tails and total quantile" begin
    d = TDist(5.0, 0.0, 1.0)
    xmax = floatmax(Float64)
    @test isfinite(logdensityof(d, 1e150))
    @test cdf(d, -1e50) > 0
    @test ccdf(d, 1e50) > 0
    @test logcdf(d, -1e50) ≈ log(cdf(d, -1e50))

    # Both log tails stay finite well past where the tails themselves underflow.
    @test cdf(d, -1e150) == 0
    @test isfinite(logcdf(d, -1e150))
    @test isfinite(logccdf(d, 1e150))

    # Squaring the standardized value would overflow at `floatmax`; `hypot` does not.
    @test isfinite(logdensityof(d, xmax))
    @test logdensityof(d, xmax) < logdensityof(d, 1e150)
    @test cdf(d, -xmax) == 0
    @test cdf(d, xmax) == 1

    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 1.0) == Inf
    @test isfinite(quantile(d, 1e-300))
    @test isfinite(quantile(d, prevfloat(1.0)))
    @test isfinite(logcdf(d, quantile(d, 1e-300)))
    @test cdf(d, quantile(d, 1e-300)) ≈ 1e-300 rtol = 1e-9
end

@testset "quantile types" begin
    #=
      An out-of-range `p` must widen to the parameters like any other argument. Typing
      the `NaN` by `p` alone drops them and makes the return type a `Union`.
    =#
    @test quantile(TDist(5.0, 0.0, 1.0), 1.5f0) isa Float64
    @test quantile(TDist(big(5.0), big(0.0), big(1.0)), 1.5) isa BigFloat
    @test quantile(TDist(5.0f0, 0.0f0, 1.0f0), 0.25f0) isa Float32
    @test Base.promote_op(quantile, TDist{Float64,Float64,Float64}, Float32) === Float64
    @test isnan(quantile(TDist(5.0, 0.0, 1.0), 1.5))
    @test isnan(quantile(TDist(5.0, 0.0, 1.0), -0.5))
end

@testset "high precision inverts the CDF exactly" begin
    setprecision(BigFloat, 256) do
        d = TDist(big(5), big(0), big(1))
        for p in (big"1e-40", big"0.25", big"0.5", 1 - big"1e-20")
            @test abs(cdf(d, quantile(d, p)) - p) < 1e-60
        end
    end
end

@testset "sampling" begin
    d = TDist(5.0, -2.5, 0.5)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), TDist(5.0f0, 0.0f0, 1.0f0)) isa Float32
    @test length(rand(Xoshiro(1), d, 7)) == 7

    # Inverting the CDF means a draw is the quantile of the underlying uniform.
    @test rand(Xoshiro(3), d) == quantile(d, rand(Xoshiro(3), Float64))

    #=
      The draw is `μ + σ z`, so its derivatives in the location and scale are one and
      the standardized draw. It has no derivative in `ν`, which enters through the
      incomplete-beta inverse.
    =#
    z = quantile(TDist(5.0, 0.0, 1.0), rand(Xoshiro(3), Float64))
    @test ForwardDiff.derivative(m -> rand(Xoshiro(3), TDist(5.0, m, 0.5)), -2.5) == 1.0
    @test ForwardDiff.derivative(s -> rand(Xoshiro(3), TDist(5.0, -2.5, s)), 0.5) ≈ z
end

@testset "show" begin
    @test string(TDist(5.0, -2.5, 0.5)) == "TDist(ν=5.0, μ=-2.5, σ=0.5)"
end
