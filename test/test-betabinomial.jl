using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using SpecialFunctions: digamma
using Test

@testset "conformance" begin
    #=
      The reference loses precision when given `Float32` shapes, by about `4e-7` on
      `BetaBinomial(4, 0.5f0, 1.5f0)`, so widen them first. This package promotes the
      shapes to the type of the evaluation point, which is `Float64` here.
    =#
    function reference_logpdf(m, x)
        r = Distributions.BetaBinomial(m.n, Float64(m.α), Float64(m.β))
        return Distributions.logpdf(r, Int(x))
    end
    measures = (
        BetaBinomial(1, 1.0, 1.0),
        BetaBinomial(5, 2.0, 3.0),
        BetaBinomial(4, 0.5f0, 1.5f0),
        BetaBinomial(20, 0.8, 2.4),
    )
    for d in measures
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = BetaBinomial(5, 2.0, 3.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "BetaBinomial(n=5, α=2.0, β=3.0)"
    @test params(d) === (n=5, α=2.0, β=3.0)
end

@testset "the trial count stays fixed" begin
    # `n` stays an integer because it sets the support and loop lengths.
    @test typeof(BetaBinomial(5, 2.0, 3.0)) === BetaBinomial{Int,Float64,Float64}
    @test typeof(BetaBinomial(Int32(5), 2.0f0, 3.0f0)) ===
        BetaBinomial{Int32,Float32,Float32}
    @test typeof(BetaBinomial(5, 1//2, 3//2)) ===
        BetaBinomial{Int,Rational{Int},Rational{Int}}

    @test eltype(BetaBinomial(5, 2.0f0, 3.0f0)) === Float32
    @test eltype(BetaBinomial(5, 2.0f0, 3.0)) === Float64
    @test eltype(BetaBinomial(5, 1//2, 3//2)) === Float64
    @test isbits(BetaBinomial(5, 2.0, 3.0))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(BetaBinomial(3, 1//2, 3//2), 1.0f0) isa Float32
    @test logdensityof(BetaBinomial(3, 1//2, 3//2), big"1.0") isa BigFloat

    # Unit shapes spread the mass evenly, so the exact value is `-log(n + 1)`.
    exact = logdensityof(BetaBinomial(3, 1, 1), big"1.0")
    @test abs(exact + log(big"4.0")) < 1e-70
end

@testset "construction never validates" begin
    negative = BetaBinomial(-1, 2.0, 3.0)
    @test !checkparams(negative)
    @test logdensityof(negative, 0.0) == -Inf

    # An invalid shape leaves no outcome finite, unlike a binomial's invalid `p`.
    for d in (
        BetaBinomial(3, -1.0, 3.0),
        BetaBinomial(3, 0.0, 3.0),
        BetaBinomial(3, 2.0, -1.0),
        BetaBinomial(3, 2.0, 0.0),
        BetaBinomial(3, Inf, 3.0),
        BetaBinomial(3, 2.0, NaN),
    )
        @test !checkparams(d)
        @test all(k -> isnan(logdensityof(d, float(k))), 0:3)
    end

    @test_throws DomainError validateparams(BetaBinomial(3, -1.0, 3.0))
    @test validateparams(BetaBinomial(3, 2.0, 3.0)) === BetaBinomial(3, 2.0, 3.0)
end

@testset "support" begin
    d = BetaBinomial(5, 2.0, 3.0)
    @test support(d) === IntegerRange(0, 5)
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === 5

    @test insupport(d, 0.0)
    @test insupport(d, 5.0)
    @test insupport(d, 3)
    @test !insupport(d, -1.0)
    @test !insupport(d, 6.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)

    @test support(BetaBinomial(-1, 2.0, 3.0)) === IntegerRange(0, -1)
end

@testset "density is total off the support" begin
    d = BetaBinomial(5, 2.0, 3.0)
    # A count outside the support makes one beta argument non-positive, where
    # `loggamma` would throw.
    for x in (-1.0, -1.5, 6.0, 2.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for k in 0:5
        @test isfinite(logdensityof(d, float(k)))
    end
end

@testset "unit shapes spread the mass evenly" begin
    for n in (0, 1, 5)
        d = BetaBinomial(n, 1.0, 1.0)
        for k in 0:n
            @test densityof(d, float(k)) ≈ 1 / (n + 1)
        end
        @test entropy(d) ≈ log(n + 1)
    end
end

@testset "n = 1 is Bernoulli" begin
    for (α, β) in ((1.0, 1.0), (2.0, 3.0), (0.5, 0.25))
        d, b = BetaBinomial(1, α, β), Bernoulli(α / (α + β))
        for x in (0.0, 1.0, 2.0, -1.0, 0.5)
            @test logdensityof(d, x) ≈ logdensityof(b, x)
        end
        @test cdf(d, 0.0) ≈ cdf(b, 0.0)
        @test mean(d) ≈ mean(b)
        @test var(d) ≈ var(b)
        @test entropy(d) ≈ entropy(b)
    end
end

@testset "concentrated shapes approach a binomial" begin
    # The variance ratio is `(α + β + n) / (α + β + 1)`, so it decays like `1 / (α + β)`.
    n, p = 5, 0.3
    for s in (1e4, 1e6)
        d, b = BetaBinomial(n, s * p, s * (1 - p)), Binomial(n, p)
        @test mean(d) ≈ mean(b)
        @test var(d) ≈ var(b) rtol = 1e-3
        for k in 0:n
            @test logdensityof(d, float(k)) ≈ logdensityof(b, float(k)) rtol = 1e-3
        end
    end
end

@testset "degenerate parameters" begin
    # No trials puts all the mass on zero successes, whatever the shapes.
    d = BetaBinomial(0, 2.0, 3.0)
    @test checkparams(d)
    @test support(d) === IntegerRange(0, 0)
    @test logdensityof(d, 0.0) == 0.0
    @test logdensityof(d, 1.0) == -Inf
    @test mean(d) == 0.0
    @test var(d) == 0.0
    @test entropy(d) == 0.0
    @test all(==(0.0), rand(Xoshiro(1), d, 8))
end

@testset "reference numerics against Distributions.jl" begin
    for (n, α, β) in ((1, 1.0, 1.0), (5, 2.0, 3.0), (10, 0.5, 0.5), (20, 0.8, 2.4))
        d, r = BetaBinomial(n, α, β), Distributions.BetaBinomial(n, α, β)
        for k in 0:n
            x = float(k)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test densityof(d, x) ≈ Distributions.pdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            # Relative error is not useful when these log values are near zero.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k) atol = 1e-12
        end
        for q in (0.0, 0.01, 0.25, 0.5, 0.9, 0.999, 1.0)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the atoms" begin
    d = BetaBinomial(5, 2.0, 3.0)
    # The CDF is constant between the atoms.
    @test cdf(d, 2.0) == cdf(d, 2.999)
    @test cdf(d, -0.5) == 0.0
    @test cdf(d, 5.0) ≈ 1.0
    @test cdf(d, 10.0) ≈ 1.0
    @test logcdf(d, -1.0) == -Inf
    @test logccdf(d, 5.0) == -Inf

    # Each tail is summed on its own, so the two agree without cancelling.
    for k in 0:5
        @test cdf(d, float(k)) + ccdf(d, float(k)) ≈ 1.0
    end

    @test [quantile(d, cdf(d, float(k))) for k in 0:5] == float.(0:5)

    # Out-of-range probabilities still return an atom.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "log-density gradient with respect to the shapes" begin
    n = 5
    for (α, β) in ((2.0, 3.0), (0.5, 0.25)), k in 0:n
        g = ForwardDiff.gradient([α, β]) do θ
            logdensityof(BetaBinomial(n, θ[1], θ[2]), float(k))
        end
        s = digamma(α + β) - digamma(n + α + β)
        @test g[1] ≈ digamma(k + α) - digamma(α) + s
        @test g[2] ≈ digamma(n - k + β) - digamma(β) + s
    end
end

@testset "sample derivative is zero" begin
    # A sample changes in steps as the shapes change.
    g = ForwardDiff.gradient([2.0, 3.0]) do θ
        rand(Xoshiro(7), BetaBinomial(5, θ[1], θ[2]))
    end
    @test iszero(g)
end

@testset "sampling" begin
    d = BetaBinomial(5, 2.0, 3.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), BetaBinomial(5, 2.0f0, 3.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 200_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 0.02
    @test var(draws) ≈ var(d) atol = 0.03
end
