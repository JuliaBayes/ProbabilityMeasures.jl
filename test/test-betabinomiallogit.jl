using ProbabilityMeasures
using ProbabilityMeasures: betashapes, logistic
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using SpecialFunctions: digamma
using Test

@testset "conformance" begin
    # The reference loses precision when given `Float32` shapes, so widen them first.
    function reference_logpdf(m, x)
        α, β = betashapes(m)
        r = Distributions.BetaBinomial(m.n, Float64(α), Float64(β))
        return Distributions.logpdf(r, Int(x))
    end
    measures = (
        BetaBinomialLogit(1, 0.0, 2.0),
        BetaBinomialLogit(5, 0.4, 4.0),
        BetaBinomialLogit(4, -0.5f0, 3.0f0),
        BetaBinomialLogit(20, -1.2, 0.8),
    )
    for d in measures
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = BetaBinomialLogit(5, 0.4, 4.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "BetaBinomialLogit(n=5, η=0.4, ϕ=4.0)"
    @test params(d) === (n=5, η=0.4, ϕ=4.0)
end

@testset "the trial count stays fixed" begin
    # `n` stays an integer because it sets the support and loop lengths.
    @test typeof(BetaBinomialLogit(5, 0.4, 4.0)) === BetaBinomialLogit{Int,Float64,Float64}
    @test typeof(BetaBinomialLogit(Int32(5), 0.4f0, 4.0f0)) ===
        BetaBinomialLogit{Int32,Float32,Float32}

    @test eltype(BetaBinomialLogit(5, 0.4f0, 4.0f0)) === Float32
    @test eltype(BetaBinomialLogit(5, 0.4f0, 4.0)) === Float64
    @test eltype(BetaBinomialLogit(5, 1, 4)) === Float64
    @test isbits(BetaBinomialLogit(5, 0.4, 4.0))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(BetaBinomialLogit(3, 1, 4), 1.0f0) isa Float32
    @test logdensityof(BetaBinomialLogit(3, 1, 4), big"1.0") isa BigFloat

    # The shapes are formed in the argument's type, so an exact `η` and `ϕ` keep it.
    exact = logdensityof(BetaBinomialLogit(3, 0, 2), big"1.0")
    @test abs(exact + log(big"4.0")) < 1e-70
end

@testset "the shapes sum to the precision" begin
    for (η, ϕ) in ((0.0, 2.0), (0.4, 4.0), (-1.2, 0.8), (3.0, 100.0))
        α, β = betashapes(BetaBinomialLogit(5, η, ϕ))
        @test α + β ≈ ϕ
        @test α / (α + β) ≈ logistic(η)
    end
end

@testset "matches the shape parameterization" begin
    for (n, η, ϕ) in ((1, 0.0, 2.0), (5, 0.4, 4.0), (20, -1.2, 0.8))
        d = BetaBinomialLogit(n, η, ϕ)
        b = BetaBinomial(n, betashapes(d)...)
        for k in 0:n
            @test logdensityof(d, float(k)) == logdensityof(b, float(k))
            @test cdf(d, float(k)) == cdf(b, float(k))
            @test ccdf(d, float(k)) == ccdf(b, float(k))
        end
        @test mean(d) ≈ mean(b)
        @test var(d) ≈ var(b)
        @test entropy(d) == entropy(b)
        @test quantile(d, 0.3) == quantile(b, 0.3)
        @test rand(Xoshiro(1), d) == rand(Xoshiro(1), b)
    end
end

@testset "construction never validates" begin
    negative = BetaBinomialLogit(-1, 0.4, 4.0)
    @test !checkparams(negative)
    @test logdensityof(negative, 0.0) == -Inf

    for d in (
        BetaBinomialLogit(3, 0.4, -1.0),
        BetaBinomialLogit(3, 0.4, 0.0),
        BetaBinomialLogit(3, 0.4, Inf),
        BetaBinomialLogit(3, 0.4, NaN),
        BetaBinomialLogit(3, NaN, 4.0),
        BetaBinomialLogit(3, Inf, 4.0),
    )
        @test !checkparams(d)
        @test all(k -> isnan(logdensityof(d, float(k))), 0:3)
    end

    @test_throws DomainError validateparams(BetaBinomialLogit(3, 0.4, -1.0))
    @test validateparams(BetaBinomialLogit(3, 0.4, 4.0)) === BetaBinomialLogit(3, 0.4, 4.0)
end

@testset "an extreme logit leaves no usable shape" begin
    #=
      `exp(η)` overflows past these magnitudes, which sends one shape to zero. The
      bound comes from the element type alone, not from `ϕ`.
    =#
    for ϕ in (1e-3, 1.0, 1e3)
        @test checkparams(BetaBinomialLogit(3, 709.0, ϕ))
        @test checkparams(BetaBinomialLogit(3, -709.0, ϕ))
        @test !checkparams(BetaBinomialLogit(3, 710.0, ϕ))
        @test !checkparams(BetaBinomialLogit(3, -710.0, ϕ))

        @test checkparams(BetaBinomialLogit(3, 88.0f0, Float32(ϕ)))
        @test !checkparams(BetaBinomialLogit(3, 89.0f0, Float32(ϕ)))
    end

    @test all(k -> isnan(logdensityof(BetaBinomialLogit(3, 800.0, 1.0), float(k))), 0:3)
end

@testset "support" begin
    d = BetaBinomialLogit(5, 0.4, 4.0)
    @test support(d) === IntegerRange(0, 5)
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === 5

    @test insupport(d, 0.0)
    @test insupport(d, 5.0)
    @test !insupport(d, -1.0)
    @test !insupport(d, 6.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)

    @test support(BetaBinomialLogit(-1, 0.4, 4.0)) === IntegerRange(0, -1)
end

@testset "density is total off the support" begin
    d = BetaBinomialLogit(5, 0.4, 4.0)
    for x in (-1.0, -1.5, 6.0, 2.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for k in 0:5
        @test isfinite(logdensityof(d, float(k)))
    end
end

@testset "a zero logit with precision two spreads the mass evenly" begin
    # Both shapes are one there, which is the uniform measure on the support.
    for n in (0, 1, 5)
        d = BetaBinomialLogit(n, 0.0, 2.0)
        for k in 0:n
            @test densityof(d, float(k)) ≈ 1 / (n + 1)
        end
        @test entropy(d) ≈ log(n + 1)
    end
end

@testset "high precision approaches a binomial" begin
    # The variance ratio is `(n + ϕ) / (1 + ϕ)`, so it decays like `1 / ϕ`.
    n, η = 5, -0.8
    for ϕ in (1e4, 1e6)
        d, b = BetaBinomialLogit(n, η, ϕ), Binomial(n, logistic(η))
        @test mean(d) ≈ mean(b)
        @test var(d) ≈ var(b) rtol = 1e-3
        for k in 0:n
            @test logdensityof(d, float(k)) ≈ logdensityof(b, float(k)) rtol = 1e-3
        end
    end
end

@testset "degenerate parameters" begin
    # No trials puts all the mass on zero successes, whatever the shapes.
    d = BetaBinomialLogit(0, 0.4, 4.0)
    @test checkparams(d)
    @test logdensityof(d, 0.0) == 0.0
    @test logdensityof(d, 1.0) == -Inf
    @test mean(d) == 0.0
    @test var(d) == 0.0
    @test entropy(d) == 0.0
    @test all(==(0.0), rand(Xoshiro(1), d, 8))
end

@testset "reference numerics against Distributions.jl" begin
    for (n, η, ϕ) in ((1, 0.0, 2.0), (5, 0.4, 4.0), (10, -0.5, 1.5), (20, -1.2, 0.8))
        d = BetaBinomialLogit(n, η, ϕ)
        r = Distributions.BetaBinomial(n, betashapes(d)...)
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
    d = BetaBinomialLogit(5, 0.4, 4.0)
    # The CDF is constant between the atoms.
    @test cdf(d, 2.0) == cdf(d, 2.999)
    @test cdf(d, -0.5) == 0.0
    @test cdf(d, 5.0) ≈ 1.0
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

@testset "log-density gradient with respect to the logit and the precision" begin
    n = 5
    for (η, ϕ) in ((0.4, 4.0), (-1.2, 0.8)), k in 0:n
        g = ForwardDiff.gradient([η, ϕ]) do θ
            logdensityof(BetaBinomialLogit(n, θ[1], θ[2]), float(k))
        end
        # The shape gradients, carried through `α = ϕσ(η)` and `β = ϕσ(-η)`.
        p, q = logistic(η), logistic(-η)
        α, β = ϕ * p, ϕ * q
        s = digamma(α + β) - digamma(n + α + β)
        dα = digamma(k + α) - digamma(α) + s
        dβ = digamma(n - k + β) - digamma(β) + s
        @test g[1] ≈ (dα - dβ) * ϕ * p * q
        @test g[2] ≈ dα * p + dβ * q
    end
end

@testset "sample derivative is zero" begin
    # A sample changes in steps as the parameters change.
    g = ForwardDiff.gradient([0.4, 4.0]) do θ
        rand(Xoshiro(7), BetaBinomialLogit(5, θ[1], θ[2]))
    end
    @test iszero(g)
end

@testset "sampling" begin
    d = BetaBinomialLogit(5, 0.4, 4.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), BetaBinomialLogit(5, 0.4f0, 4.0f0)) isa Float32
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
