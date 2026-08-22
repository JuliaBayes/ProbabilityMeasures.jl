using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using Random: Xoshiro
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Geometric(m.p), Int(x))
    for d in (Geometric(0.3), Geometric(0.6f0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Geometric(0.3)
    @test d isa DiscreteUnivariateMeasure
    @test string(d) == "Geometric(p=0.3)"
    @test params(d) === (p=0.3,)
    @test Geometric() === Geometric(0.5)
    @test eltype(Geometric(0.3f0)) === Float32
end

@testset "parameters" begin
    @test checkparams(Geometric(0.3))
    @test checkparams(Geometric(1.0))
    @test !checkparams(Geometric(1.1))
    @test_throws DomainError validateparams(Geometric(0.0))
    @test validateparams(Geometric(0.25)) === Geometric(0.25)
end

@testset "support" begin
    d = Geometric(0.3)
    @test support(d) === NonNegativeIntegers()
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === Inf
    @test insupport(d, 0.0)
    @test insupport(d, 3)
    @test !insupport(d, -1.0)
    @test !insupport(d, 0.5)
    @test !insupport(d, Inf)
    @test !insupport(d, NaN)
end

@testset "reference numerics" begin
    for p in (0.1, 0.3, 0.8)
        d, r = Geometric(p), Distributions.Geometric(p)
        for x in (0.0, 1.0, 5.0, 10.0)
            @test cdf(d, x) ≈ Distributions.cdf(r, Int(x))
            @test ccdf(d, x) ≈ Distributions.ccdf(r, Int(x))
            @test logcdf(d, x) ≈ Distributions.logcdf(r, Int(x))
            @test logccdf(d, x) ≈ Distributions.logccdf(r, Int(x))
        end
        for q in (0.0, 0.1, 0.5, 0.99)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "endpoints and tails" begin
    d = Geometric(0.3)
    @test quantile(d, 1.0) == Inf
    @test isfinite(logccdf(d, 3000.0))
    @test ccdf(d, 3000.0) == 0.0

    pointmass = Geometric(1.0)
    @test logdensityof(pointmass, 0.0) == 0.0
    @test logdensityof(pointmass, 1.0) == -Inf
    @test cdf(pointmass, 0.0) == 1.0
    @test ccdf(pointmass, 0.0) == 0.0
    @test quantile(pointmass, 0.0) == 0.0
    @test quantile(pointmass, 1.0) == 0.0
    @test mean(pointmass) == 0.0
    @test var(pointmass) == 0.0
    @test entropy(pointmass) == 0.0
    @test all(iszero, rand(Xoshiro(0x524549), pointmass, 8))
end

@testset "total off the domains" begin
    d = Geometric(0.3)
    @test cdf(d, -1.0) == 0.0
    @test ccdf(d, -1.0) == 1.0
    @test logcdf(d, -1.0) == -Inf
    @test logccdf(d, -1.0) == 0.0
    @test cdf(d, 1.5) == cdf(d, 1.0)
    for x in (-1.0, 0.5, Inf, -Inf, NaN)
        @test logdensityof(d, x) == -Inf
    end
end
