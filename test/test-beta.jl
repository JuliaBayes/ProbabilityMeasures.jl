using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using QuadGK: quadgk
using Random: Random, Xoshiro
using SpecialFunctions: trigamma
using Test

reference(d) = Distributions.Beta(d.α, d.β)

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(reference(m), x)
    for d in (Beta(2.0, 3.0), Beta(1.0, 1.0), Beta(0.5, 0.5), Beta(2.0f0, 5.0f0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Beta(2.0, 3.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Continuous}
    @test d isa ContinuousUnivariateMeasure
    @test !(d isa DiscreteUnivariateMeasure)
    @test string(d) == "Beta(α=2.0, β=3.0)"
    @test params(d) === (α=2.0, β=3.0)
    @test Beta() === Beta(1.0, 1.0)
end

@testset "numeric types" begin
    @test typeof(Beta(2.0, 3.0)) === Beta{Float64,Float64}
    @test typeof(Beta(2.0f0, 3.0f0)) === Beta{Float32,Float32}
    @test typeof(Beta(2, 3)) === Beta{Int,Int}
    @test typeof(Beta(1//2, 3)) === Beta{Rational{Int},Int}

    @test eltype(Beta(2.0f0, 3.0f0)) === Float32
    @test eltype(Beta(2, 3)) === Float64
    @test isbits(Beta(2.0, 3.0))
end

@testset "precision follows the argument, not the parameter" begin
    @test logdensityof(Beta(2, 3), 0.5f0) isa Float32
    @test logdensityof(Beta(2, 3), big"0.5") isa BigFloat

    # `B(2, 3) = 1/12`, so the density at one half is `12 * (1/2) * (1/4)`.
    exact = logdensityof(Beta(2, 3), big"0.5")
    @test abs(exact - log(big"12" / 8)) < 1e-70
end

@testset "construction never validates" begin
    for bad in
        (Beta(0.0, 1.0), Beta(1.0, 0.0), Beta(-1.0, 1.0), Beta(Inf, 1.0), Beta(1.0, NaN))
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, 0.5))
        @test_throws DomainError validateparams(bad)
    end

    @test checkparams(Beta(2.0, 3.0))
    @test validateparams(Beta(2.0, 3.0)) === Beta(2.0, 3.0)
end

@testset "support" begin
    d = Beta(2.0, 3.0)
    @test support(d) === UnitInterval()
    @test minimum(support(d)) === 0.0
    @test maximum(support(d)) === 1.0
    @test isbits(support(d))

    @test insupport(d, 0.0)
    @test insupport(d, 1.0)
    @test insupport(d, 0.5)
    @test !insupport(d, -0.001)
    @test !insupport(d, 1.001)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)
end

@testset "density is total off the support" begin
    d = Beta(2.0, 3.0)
    for x in (-0.5, 1.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for x in 0.05:0.05:0.95
        @test isfinite(logdensityof(d, x))
    end
end

@testset "a shape below one puts an infinite density at the endpoint" begin
    d = Beta(0.5, 0.5)
    @test logdensityof(d, 0.0) == Inf
    @test logdensityof(d, 1.0) == Inf
    # A unit shape leaves the endpoint finite rather than producing `0 * -Inf`.
    @test logdensityof(Beta(1.0, 2.0), 0.0) ≈ log(2.0)
    @test logdensityof(Beta(2.0, 1.0), 1.0) ≈ log(2.0)
    @test logdensityof(Beta(1.0, 1.0), 0.0) == 0.0
    @test logdensityof(Beta(1.0, 1.0), 1.0) == 0.0
end

@testset "normalization" begin
    for (α, β) in ((2.0, 3.0), (1.0, 1.0), (0.5, 0.5), (5.0, 1.0), (7.0, 11.0))
        total, err = quadgk(x -> densityof(Beta(α, β), x), 0.0, 1.0; rtol=1e-10)
        @test total ≈ 1 atol = max(1e-9, 10err)
    end
end

@testset "reference numerics against Distributions.jl" begin
    for (α, β) in ((2.0, 3.0), (1.0, 1.0), (0.5, 0.5), (5.0, 1.0), (7.0, 11.0))
        d, r = Beta(α, β), Distributions.Beta(α, β)
        for x in 0.02:0.02:0.98
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
            @test cdf(d, x) ≈ Distributions.cdf(r, x)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
            @test logcdf(d, x) ≈ Distributions.logcdf(r, x) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, x) atol = 1e-12
        end
        for q in (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)
            @test quantile(d, q) ≈ Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test median(d) ≈ Distributions.median(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "invalid probabilities do not throw" begin
    d = Beta(2.0, 3.0)
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test isnan(quantile(d, q))
        @test isnan(quantile(d, ForwardDiff.Dual(q, 1.0)))
    end
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == 1.0
end

@testset "the fixed-length path matches the converging one" begin
    # Both tails, on both sides of the crossover.
    for a in (0.1, 1.0, 7.5, 100.0, 1000.0), b in (0.5, 3.0, 300.0)
        for x in (1e-6, 0.1, 0.5, 0.9, 1 - 1e-6)
            lp, lq = ProbabilityMeasures.logbetainc(a, b, x, 1 - x)
            sp, sq = ProbabilityMeasures.logbetainc_fixed(a, b, x, 1 - x)
            @test sp ≈ lp atol = 1e-12 rtol = 1e-12
            @test sq ≈ lq atol = 1e-12 rtol = 1e-12
        end
    end

    # The fixed number of Newton steps reaches the converging answer for every noise value.
    # With plain floats the steps still call the converging tails; the composition with
    # the fixed tails runs under Reactant in `test/reactant`.
    rng = Xoshiro(3)
    for (a, b) in ((0.05, 3.0), (0.5, 0.5), (2.0, 3.0), (100.0, 0.3), (300.0, 500.0))
        for _ in 1:1_000
            p = rand(rng)
            @test ProbabilityMeasures.betaquantile_fixed(a, b, p) ≈
                ProbabilityMeasures.betaquantile(a, b, p) rtol = 1e-11
        end
    end
end

@testset "the continued fraction keeps its type" begin
    for T in (Float32, Float64, BigFloat)
        d = Beta(T(2), T(3))
        @test cdf(d, T(1) / 4) isa T
        @test ccdf(d, T(1) / 4) isa T
        @test quantile(d, T(1) / 4) isa T
        @test rand(Xoshiro(1), d) isa T
    end
    widened = cdf(Beta(big"2.0", big"3.0"), big"0.25")
    @test widened ≈ cdf(Beta(2.0, 3.0), 0.25) atol = 1e-14
end

@testset "the CDF inverts at high precision" begin
    setprecision(BigFloat, 256) do
        d = Beta(big"2.0", big"3.0")
        for p in (big"0.01", big"0.25", big"0.5", big"0.9")
            @test abs(cdf(d, quantile(d, p)) - p) < 1e-60
        end
    end
end

@testset "log-density gradient with respect to the shapes" begin
    ψ = ProbabilityMeasures.digamma
    for (α, β) in ((2.0, 3.0), (0.5, 0.5), (7.0, 11.0)), x in (0.1, 0.5, 0.9)
        g = ForwardDiff.gradient(p -> logdensityof(Beta(p[1], p[2]), x), [α, β])
        @test g[1] ≈ log(x) - ψ(α) + ψ(α + β)
        @test g[2] ≈ log1p(-x) - ψ(β) + ψ(α + β)
    end
end

@testset "the sample derivative is the reparameterization gradient" begin
    #=
      A draw is a smooth function of the noise and the shapes, so its derivative at fixed
      noise averages to the derivative of the mean, `β / (α + β)²`, and its log to the
      derivative of `E[log x] = ψ(α) - ψ(α + β)`.
    =#
    n = 100_000
    for (α, β) in ((0.5, 0.5), (2.0, 3.0), (0.05, 3.0))
        rng = Xoshiro(7)
        dmean, dlog = 0.0, 0.0
        for _ in 1:n
            seed = rand(rng, UInt)
            f = a -> rand(Xoshiro(seed), Beta(a, β))
            x = f(α)
            dx = ForwardDiff.derivative(f, α)
            dmean += dx
            dlog += dx / x
        end
        @test dmean / n ≈ β / (α + β)^2 rtol = 0.03
        @test dlog / n ≈ trigamma(α) - trigamma(α + β) rtol = 0.03
    end
end

@testset "sampling" begin
    d = Beta(2.0, 3.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Beta(2.0f0, 3.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 100_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 5 * std(d) / sqrt(100_000)
    @test var(draws) ≈ var(d) rtol = 0.02

    # Tiny shapes put nearly all the mass at the endpoints; the log-space ratio keeps
    # every draw finite and on the interval.
    small = rand(Xoshiro(1), Beta(0.01, 0.02), 10_000)
    @test all(x -> insupport(Beta(0.01, 0.02), x), small)

    # Invalid parameters give `NaN` rather than throwing.
    for m in (Beta(-1.0, 1.0), Beta(NaN, 1.0), Beta(1.0, 0.0))
        @test isnan(rand(Xoshiro(1), m))
    end
end
