using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    function reference_logpdf(m, x)
        return Distributions.logpdf(Distributions.Poisson(m.λ), Int(x))
    end
    for d in (Poisson(0.5), Poisson(4.0), Poisson(2.5f0), Poisson(11.0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Poisson(4.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "Poisson(λ=4.0)"
    @test params(d) === (λ=4.0,)
end

@testset "numeric types" begin
    @test typeof(Poisson(4.0)) === Poisson{Float64}
    @test typeof(Poisson(4.0f0)) === Poisson{Float32}
    @test typeof(Poisson(2)) === Poisson{Int}
    @test typeof(Poisson(5//2)) === Poisson{Rational{Int}}

    @test eltype(Poisson(4.0f0)) === Float32
    @test eltype(Poisson(2)) === Float64
    @test isbits(Poisson(4.0))
end

@testset "precision follows the argument, not the parameter" begin
    @test logdensityof(Poisson(2), 1.0f0) isa Float32
    @test logdensityof(Poisson(2), big"1.0") isa BigFloat

    # An integer rate must not drop the argument back to `Float64`.
    exact = logdensityof(Poisson(2), big"1.0")
    @test abs(exact - (log(big"2.0") - 2)) < 1e-70
end

@testset "construction never validates" begin
    # A negative rate leaves `k = 0` finite, where the offending factor drops out.
    negative = Poisson(-1.0)
    @test !checkparams(negative)
    @test isfinite(logdensityof(negative, 0.0))
    @test isnan(logdensityof(negative, 1.0))

    @test !checkparams(Poisson(NaN))
    @test !checkparams(Poisson(Inf))
    @test !isfinite(logdensityof(Poisson(NaN), 0.0))
    @test !isfinite(logdensityof(Poisson(Inf), 0.0))

    @test_throws DomainError validateparams(Poisson(-1.0))
    @test validateparams(Poisson(4.0)) === Poisson(4.0)
end

@testset "support" begin
    d = Poisson(4.0)
    @test support(d) === NonNegativeIntegers()
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === Inf

    @test insupport(d, 0.0)
    @test insupport(d, 7)
    @test insupport(d, 1e6)
    @test !insupport(d, -1.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)
end

@testset "density is total off the support" begin
    d = Poisson(4.0)
    # The count is clamped before `loggamma`, so values below zero do not throw.
    for x in (-1.0, -1.5, 2.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for k in 0:20
        @test isfinite(logdensityof(d, float(k)))
    end
end

@testset "a zero rate puts all the mass on zero" begin
    d = Poisson(0.0)
    @test checkparams(d)
    # The zero term must win over `log(0)`.
    @test logdensityof(d, 0.0) == 0.0
    @test logdensityof(d, 1.0) == -Inf
    @test mean(d) == 0.0
    @test var(d) == 0.0
    @test entropy(d) == 0.0
    @test cdf(d, 0.0) == 1.0
    @test ccdf(d, 0.0) == 0.0
    @test all(==(0.0), rand(Xoshiro(1), d, 8))
end

@testset "normalization" begin
    # The support has no upper end, so sum far enough out to cover it.
    for λ in (0.5, 1.0, 4.0, 11.0, 40.0)
        @test sum(k -> densityof(Poisson(λ), float(k)), 0:400) ≈ 1
    end
end

@testset "reference numerics against Distributions.jl" begin
    for λ in (0.1, 0.5, 1.0, 4.0, 11.0, 40.0)
        d, r = Poisson(λ), Distributions.Poisson(λ)
        for k in 0:Distributions.quantile(r, 0.9999)
            x = float(k)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test densityof(d, x) ≈ Distributions.pdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            # Relative error is not useful when these log values are near zero.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k) atol = 1e-12
        end
        for q in (0.01, 0.25, 0.5, 0.9, 0.999)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the atoms" begin
    d = Poisson(4.0)
    # The CDF is constant between the atoms.
    @test cdf(d, 2.0) == cdf(d, 2.999)
    @test cdf(d, -0.5) == 0.0
    # A summed tail reaches one only up to rounding.
    @test ccdf(d, -0.5) ≈ 1.0
    @test logcdf(d, -1.0) == -Inf

    # Each tail is summed on its own, so the two agree without cancelling.
    for k in 0:20
        @test cdf(d, float(k)) + ccdf(d, float(k)) ≈ 1.0
    end

    # Inverting the CDF recovers each atom while the two sides still differ.
    ks = [k for k in 0:20 if cdf(d, float(k)) < 1.0]
    @test [quantile(d, cdf(d, float(k))) for k in ks] == float.(ks)

    # Out-of-range probabilities still return an outcome in the support.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "sums stop above the mean" begin
    d = Poisson(4.0)
    # The horizon covers the whole `Float64` range of the CDF.
    @test cdf(d, float(ProbabilityMeasures.horizon(d))) ≈ 1.0
    @test ccdf(d, float(ProbabilityMeasures.horizon(d))) == 0.0
    # A probability of one has no finite answer, so the last outcome stands in.
    @test quantile(d, 1.0) == float(ProbabilityMeasures.horizon(d))

    # A rate that cannot be counted leaves one term rather than a hopeless loop.
    @test ProbabilityMeasures.horizon(Poisson(NaN)) == 0
    @test ProbabilityMeasures.horizon(Poisson(Inf)) == 0
    @test ProbabilityMeasures.horizon(Poisson(-Inf)) == 0
    @test ProbabilityMeasures.horizon(Poisson(-1e300)) == 0
end

@testset "log-density gradient with respect to the rate" begin
    for λ in (0.5, 4.0, 11.0), k in 0:12
        g = ForwardDiff.derivative(l -> logdensityof(Poisson(l), float(k)), λ)
        @test g ≈ k / λ - 1
    end
end

@testset "sample derivative is zero" begin
    # A sample changes in steps as `λ` changes.
    g = ForwardDiff.derivative(l -> rand(Xoshiro(7), Poisson(l)), 4.0)
    @test iszero(g)
end

@testset "sampling" begin
    d = Poisson(4.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Poisson(4.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 100_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 0.02
    @test var(draws) ≈ var(d) atol = 0.05

    # The draw inverts the CDF, so a large rate is not lost to underflow.
    @test all(x -> insupport(d, x), rand(Xoshiro(1), Poisson(1000.0), 8))
end
