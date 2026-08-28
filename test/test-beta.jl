using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using DifferentiationInterface: AutoForwardDiff
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using QuadGK: quadgk
using Random: Random, Xoshiro
using Test

reference(d) = Distributions.Beta(d.α, d.β)

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(reference(m), x)
    #=
      Two blocks of the suite are off. `rand` reaches `SpecialFunctions.beta_inc`,
      whose asymptotic branches allocate, so the allocation check cannot run; the
      density is checked on its own below. Only ForwardDiff is swept because Zygote and
      Mooncake rewrite Julia source rather than wrap numeric types, so they reach
      `beta_inc`, which carries no ChainRules rule and hands back a zero derivative with
      respect to the second shape. ReverseDiff is correct here but is left out so the
      swept set matches the documented support.
    =#
    for d in (Beta(2.0, 3.0), Beta(1.0, 1.0), Beta(0.5, 0.5), Beta(2.0f0, 5.0f0))
        test_measure(
            d;
            name=string(d),
            reference_logpdf=reference_logpdf,
            check_allocations=false,
            ad_backends=(AutoForwardDiff(),),
        )
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

#=
  A `Dual` is `Real` but not one of the `BetaTailFloat` types, so it takes the continued
  fraction while a plain float reaches `SpecialFunctions.beta_inc`. Their values have to
  agree.
=#
@testset "both paths agree" begin
    # A `Dual` argument sends the call down the continued fraction.
    fraction(f, d, x) = ForwardDiff.value(f(d, ForwardDiff.Dual(x, 1.0)))
    for (α, β) in ((2.0, 3.0), (1.0, 1.0), (0.5, 0.5), (5.0, 1.0), (7.0, 11.0))
        d = Beta(α, β)
        for x in 0.02:0.02:0.98
            @test fraction(cdf, d, x) ≈ cdf(d, x) atol = 1e-12
            @test fraction(ccdf, d, x) ≈ ccdf(d, x) atol = 1e-12
        end
        for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            @test fraction(quantile, d, q) ≈ quantile(d, q) atol = 1e-10
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
    # `beta_inc` has no `BigFloat` method, so this value comes from the fraction.
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

@testset "sample derivative is not zero" begin
    # A draw is the quantile of a uniform draw, so it moves with both shapes.
    for (α, β) in ((2.0, 3.0), (0.5, 0.5), (7.0, 11.0))
        g = ForwardDiff.gradient(p -> rand(Xoshiro(7), Beta(p[1], p[2])), [α, β])
        @test all(isfinite, g)
        @test any(!iszero, g)
    end

    # A draw is exactly that quantile.
    for seed in 1:20
        u = rand(Xoshiro(seed), Float64)
        @test rand(Xoshiro(seed), Beta(2.0, 3.0)) === quantile(Beta(2.0, 3.0), u)
    end
end

#=
  The suite's allocation check is off for `Beta`, so cover the density here. `rand` is
  left out on purpose: it reaches `beta_inc`, which allocates.
=#
@testset "the density does not allocate" begin
    d, x = Beta(2.0, 3.0), 0.25
    logdensityof(d, x) # compile first
    @test (@allocated logdensityof(d, x)) == 0
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
end
