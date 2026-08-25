using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.LogNormal(m.μ, m.σ), x)
    for d in (
        LogNormal(0.0, 1.0), LogNormal(-1.0, 0.5), LogNormal(2.0f0, 0.75f0), LogNormal(0, 1)
    )
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
    @test !checkparams(LogNormal(0.0, Inf))
end

@testset "positive support" begin
    d = LogNormal(0.0, 1.0)
    @test support(d) === PositiveReals()
    @test logdensityof(d, 0.0) == -Inf
    @test logdensityof(d, -1.0) == -Inf
    @test cdf(d, -1.0) == 0.0
    @test ccdf(d, -1.0) == 1.0
    @test logcdf(d, -1.0) == -Inf
    @test logccdf(d, -1.0) == 0.0
end

@testset "reference numerics against Distributions.jl" begin
    d = LogNormal(-0.5, 1.25)
    r = Distributions.LogNormal(-0.5, 1.25)
    for x in (0.1, 10.0)
        @test cdf(d, x) ≈ Distributions.cdf(r, x)
        @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
        @test logcdf(d, x) ≈ Distributions.logcdf(r, x)
        @test logccdf(d, x) ≈ Distributions.logccdf(r, x)
    end
    @test quantile(d, 0.9) ≈ Distributions.quantile(r, 0.9)
    @test mean(d) ≈ Distributions.mean(r)
    @test var(d) ≈ Distributions.var(r)
    @test entropy(d) ≈ Distributions.entropy(r)
end
