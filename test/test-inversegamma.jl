using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using SpecialFunctions: digamma
using Test

@testset "conformance" begin
    # Widen the parameters because Distributions.jl works at their own precision.
    function reference_logpdf(m, x)
        r = Distributions.InverseGamma(Float64(m.α), Float64(m.θ))
        return Distributions.logpdf(r, x)
    end
    for d in (
        InverseGamma(6.0, 1.0),
        InverseGamma(8.0, 2.5),
        InverseGamma(7.0f0, 0.5f0),
        InverseGamma(5, 2),
    )
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
    # A shape at or below two leaves the variance infinite, so the sampled moments
    # have nothing to converge to.
    for d in (InverseGamma(0.5, 1.0), InverseGamma(1.5, 2.0))
        test_measure(d; name=string(d), check_moments=false)
    end
end

@testset "traits" begin
    d = InverseGamma(3.0, 2.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Continuous}
    @test d isa ContinuousUnivariateMeasure
    @test string(d) == "InverseGamma(α=3.0, θ=2.0)"
    @test params(d) === (α=3.0, θ=2.0)
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(1.0, 1.0)
    @test typeof(InverseGamma(dual, 1.0)) === InverseGamma{typeof(dual),Float64}
    @test typeof(InverseGamma(3, 1.0)) === InverseGamma{Int,Float64}
    @test typeof(InverseGamma(3.0f0, 1.0f0)) === InverseGamma{Float32,Float32}

    # The one-argument form sets the scale to one in the shape's own type.
    @test InverseGamma(3) === InverseGamma(3, 1)

    @test eltype(InverseGamma(3, 1)) === Float64
    @test eltype(InverseGamma(3.0f0, 1.0f0)) === Float32
    @test isbits(InverseGamma(3.0, 2.0))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(InverseGamma(3, 2), 1.0f0) isa Float32
    @test logdensityof(InverseGamma(3, 2), big"1.0") isa BigFloat

    exact = logdensityof(InverseGamma(3, 2), big"1.0")
    full = logdensityof(InverseGamma(big"3.0", big"2.0"), big"1.0")
    @test abs(exact - full) < 1e-70

    @test logdensityof(InverseGamma(3, 2), 1//2) isa Float64
    @test logdensityof(InverseGamma(3, 2), -1//2) === -Inf
end

@testset "construction never validates" begin
    for d in (
        InverseGamma(-1.0, 1.0),
        InverseGamma(0.0, 1.0),
        InverseGamma(1.0, -1.0),
        InverseGamma(1.0, 0.0),
    )
        @test !checkparams(d)
        @test !isfinite(logdensityof(d, 1.0))
    end
    @test !checkparams(InverseGamma(Inf, 1.0))
    @test checkparams(InverseGamma(3.0, 2.0))

    @test_throws DomainError validateparams(InverseGamma(-1.0, 1.0))
    @test validateparams(InverseGamma(3.0, 2.0)) === InverseGamma(3.0, 2.0)
end

@testset "invalid parameters give NaN, not a partial answer" begin
    for d in (InverseGamma(-1.0, 1.0), InverseGamma(3.0, -1.0), InverseGamma(NaN, 1.0))
        @test isnan(cdf(d, 1.0))
        @test isnan(ccdf(d, 1.0))
        @test isnan(logcdf(d, 1.0))
        @test isnan(logccdf(d, 1.0))
        @test isnan(quantile(d, 0.5))
        @test isnan(entropy(d))
    end
end

@testset "support" begin
    d = InverseGamma(3.0, 2.0)
    @test support(d) === PositiveReals()
    @test insupport(d, 1e-300)
    @test insupport(d, 1e300)
    @test !insupport(d, 0.0)
    @test !insupport(d, -1.0)
    @test !insupport(d, Inf)
    @test !insupport(d, NaN)
end

@testset "density is total off the support" begin
    for d in (InverseGamma(0.5, 1.0), InverseGamma(3.0, 2.0))
        for x in (0.0, -1.0, -Inf, NaN, -floatmax(Float64))
            @test logdensityof(d, x) == -Inf
        end
        @test !isfinite(logdensityof(d, Inf))
    end
end

@testset "the reciprocal of a gamma draw" begin
    # If `X` follows `InverseGamma(α, θ)` then `1/X` follows `Gamma(α, 1/θ)`, so the
    # densities differ by the Jacobian `1/x²`.
    for α in (0.5, 1.0, 3.0, 9.0), θ in (0.5, 2.0), x in (0.1, 0.7, 3.0, 20.0)
        g = Gamma(α, 1 / θ)
        @test logdensityof(InverseGamma(α, θ), x) ≈ logdensityof(g, 1 / x) - 2 * log(x)
        @test cdf(InverseGamma(α, θ), x) ≈ ccdf(g, 1 / x)
        @test logccdf(InverseGamma(α, θ), x) ≈ logcdf(g, 1 / x)
    end
end

@testset "reference numerics against Distributions.jl" begin
    ref(α, θ) = Distributions.InverseGamma(α, θ)
    for α in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 200.0), θ in (0.5, 1.0, 4.0)
        d, r = InverseGamma(α, θ), ref(α, θ)
        for p in (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
            x = Distributions.quantile(r, p)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
            @test cdf(d, x) ≈ Distributions.cdf(r, x)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
            # Use an absolute tolerance near `log(1) == 0`.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, x) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, x) atol = 1e-12
            @test quantile(d, p) ≈ x rtol = 1e-10
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test median(d) ≈ Distributions.median(r) rtol = 1e-10
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "undefined moments are infinite" begin
    @test mean(InverseGamma(0.5, 2.0)) == Inf
    @test mean(InverseGamma(1.0, 2.0)) == Inf
    @test mean(InverseGamma(2.0, 2.0)) == 2.0
    @test var(InverseGamma(1.5, 2.0)) == Inf
    @test var(InverseGamma(2.0, 2.0)) == Inf
    @test var(InverseGamma(3.0, 2.0)) == 1.0
    @test std(InverseGamma(1.5, 2.0)) == Inf
end

@testset "log tails stay finite where the probability underflows" begin
    d = InverseGamma(3.0, 2.0)
    # `P(X > x) = P(3, 2/x)`, which behaves like `(2/x)³/6` as `x` grows.
    @test ccdf(d, 1e120) == 0.0
    @test logccdf(d, 1e120) ≈ 3 * log(2e-120) - log(6.0) rtol = 1e-6
    # The lower tail decays like `e^{-2/x}`, which underflows well before its logarithm.
    @test cdf(d, 1e-3) == 0.0
    @test isfinite(logcdf(d, 1e-3))
end

@testset "the lower tail inverts without forming 1 - p" begin
    #=
      A probability this small has no complement in floating point: `1 - p` rounds to
      one. Inverting the upper incomplete gamma integral directly keeps it.
    =#
    for α in (0.5, 3.0, 9.0), θ in (1.0, 2.5)
        d = InverseGamma(α, θ)
        for p in (1e-300, 1e-100, 1e-20, 1e-8)
            x = quantile(d, p)
            @test 0 < x < Inf
            @test cdf(d, x) ≈ p rtol = 1e-10
        end
    end
end

@testset "quantile is total and inverts the CDF" begin
    d = InverseGamma(3.0, 2.0)
    for p in (-0.001, 1.001, -Inf, Inf, NaN)
        @test isnan(quantile(d, p))
    end
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == Inf

    for α in (0.05, 0.5, 1.0, 3.0, 50.0), p in (1e-12, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-9)
        m = InverseGamma(α, 2.0)
        @test cdf(m, quantile(m, p)) ≈ p rtol = 1e-10
    end
end

@testset "the quantile keeps BigFloat precision" begin
    setprecision(BigFloat, 256) do
        d = InverseGamma(big"2.5", big"1.5")
        for p in (big"1e-40", big"0.25", big"0.5", big"0.99")
            x = quantile(d, p)
            @test x isa BigFloat
            @test abs(cdf(d, x) - p) < 1e-60 * p
        end
    end
end

@testset "distribution functions keep their type" begin
    for T in (Float32, Float64, BigFloat)
        d = InverseGamma(T(3), T(2))
        @test cdf(d, T(1)) isa T
        @test ccdf(d, T(1)) isa T
        @test logcdf(d, T(1)) isa T
        @test logccdf(d, T(1)) isa T
        @test quantile(d, T(1) / 4) isa T
        @test entropy(d) isa T
    end
end

@testset "log-density gradient with respect to the parameters" begin
    for α in (0.5, 3.0, 9.0), θ in (0.5, 2.0), x in (0.3, 1.0, 9.0)
        g = ForwardDiff.gradient([α, θ]) do p
            logdensityof(InverseGamma(p[1], p[2]), x)
        end
        @test g[1] ≈ log(θ) - digamma(α) - log(x)
        @test g[2] ≈ α / θ - 1 / x
    end
end

@testset "sampling" begin
    d = InverseGamma(3.0, 2.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), InverseGamma(3.0f0, 2.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    # Both the direct sampler and Gamma's boost below a unit shape.
    for α in (0.2, 0.9, 6.0, 40.0)
        m = InverseGamma(α, 2.0)
        draws = rand(Xoshiro(20250801), m, 200_000)
        @test all(x -> insupport(m, x), draws)
        # Compare medians: the mean is undefined for the smaller shapes.
        @test median(draws) ≈ median(m) rtol = 0.02
    end
end

@testset "sample derivative follows the parameters" begin
    for α in (0.4, 3.0, 9.0), θ in (0.5, 2.0)
        draw(p) = rand(Xoshiro(7), InverseGamma(p[1], p[2]))
        g = ForwardDiff.gradient(draw, [α, θ])
        h = 1e-6
        dα = (draw([α + h, θ]) - draw([α - h, θ])) / 2h
        dθ = (draw([α, θ + h]) - draw([α, θ - h])) / 2h
        @test g[1] ≈ dα rtol = 1e-4
        @test g[2] ≈ dθ rtol = 1e-6
        @test !iszero(g[1])
    end
end
