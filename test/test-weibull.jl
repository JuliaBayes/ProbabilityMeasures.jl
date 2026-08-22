using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Xoshiro
using Test

@testset "conformance" begin
    reference_logpdf(m, x) =
        Distributions.logpdf(Distributions.Weibull(Float64(m.α), Float64(m.θ)), x)
    for d in (Weibull(1.0, 1.0), Weibull(0.75, 2.5), Weibull(2.0f0, 0.5f0), Weibull(2, 3))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "parameters and support" begin
    dual = ForwardDiff.Dual(2.0, 1.0)
    @test typeof(Weibull(dual, 1.0)) === Weibull{typeof(dual),Float64}
    @test typeof(Weibull(2.0f0, 1)) === Weibull{Float32,Int}
    @test support(Weibull()) === NonNegativeReals()

    @test checkparams(Weibull(2.0, 3.0))
    @test !checkparams(Weibull(0.0, 3.0))
    @test !checkparams(Weibull(2.0, 0.0))
    @test !checkparams(Weibull(Inf, 3.0))
end

@testset "density at zero" begin
    @test logdensityof(Weibull(0.5, 2.0), 0.0) == Inf
    @test logdensityof(Weibull(1.0, 2.0), 0.0) == -log(2.0)
    @test logdensityof(Weibull(2.0, 2.0), 0.0) == -Inf
    @test logdensityof(Weibull(0.5, 2.0), -1.0) == -Inf
end

@testset "distribution-function tails" begin
    exponential = Weibull(1.0, 1.0)
    @test cdf(exponential, 1e-20) > 0
    @test cdf(exponential, 1e-20) ≈ 1e-20
    @test ccdf(exponential, 1000.0) == 0.0
    @test logccdf(exponential, 1000.0) == -1000.0

    for p in (1e-20, prevfloat(1.0))
        @test cdf(exponential, quantile(exponential, p)) ≈ p
    end
end

@testset "reference numerics against Distributions.jl" begin
    for (α, θ) in ((0.5, 0.4), (1.0, 1.0), (2.5, 3.0)), x in (0.2, 1.7, 8.0)
        d, r = Weibull(α, θ), Distributions.Weibull(α, θ)
        @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
        @test cdf(d, x) ≈ Distributions.cdf(r, x)
        @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
        @test logcdf(d, x) ≈ Distributions.logcdf(r, x)
        @test logccdf(d, x) ≈ Distributions.logccdf(r, x)
    end
    for (α, θ) in ((0.5, 0.4), (1.0, 1.0), (2.5, 3.0)), p in (0.01, 0.5, 0.999)
        @test quantile(Weibull(α, θ), p) ≈
            Distributions.quantile(Distributions.Weibull(α, θ), p)
    end
    for (α, θ) in ((0.5, 0.4), (1.0, 1.0), (2.5, 3.0))
        d, r = Weibull(α, θ), Distributions.Weibull(α, θ)
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "inverse-CDF sampling" begin
    d = Weibull(1.5, 2.0)
    u = rand(Xoshiro(0x4153554b41), Float64)
    @test rand(Xoshiro(0x4153554b41), d) == quantile(d, u)
end
