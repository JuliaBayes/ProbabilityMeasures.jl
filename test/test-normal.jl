using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Normal(m.μ, m.σ), x)
    for d in (Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0), Normal(0, 1))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(0.0, 1.0)
    @test typeof(Normal(dual, 1.0)) === Normal{typeof(dual),Float64}
    @test typeof(Normal(0.0f0, 1)) === Normal{Float32,Int}
    @test typeof(Normal(0.0f0, 1.0)) === Normal{Float32,Float64}

    @test Normal(0.0f0, 1.0f0).σ isa Float32
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Normal(0, 1), 1.0f0) isa Float32
    @test logdensityof(Normal(0, 1), big"1.0") isa BigFloat

    # Integer parameters must not reduce `BigFloat` precision.
    exact = logdensityof(Normal(0, 1), big"1.0")
    full = logdensityof(Normal(big"0.0", big"1.0"), big"1.0")
    @test abs(exact - full) < 1e-70
end

@testset "construction never validates" begin
    d = Normal(0.0, -1.0)
    @test !checkparams(d)
    @test isnan(logdensityof(d, 0.0))
    @test checkparams(Normal(0.0, 1.0))
    @test !checkparams(Normal(Inf, 1.0))
end

@testset "logcdf beats log(cdf) in the tail" begin
    d = Normal(0.0, 1.0)
    # The CDF underflows here, but its logarithm should remain finite.
    @test cdf(d, -40.0) == 0.0
    @test isfinite(logcdf(d, -40.0))
    @test logcdf(d, -40.0) ≈ Distributions.logcdf(Distributions.Normal(), -40.0)
    @test isfinite(logccdf(d, 40.0))
end

@testset "reference numerics against Distributions.jl" begin
    ref(μ, σ) = Distributions.Normal(μ, σ)
    for (μ, σ) in ((0.0, 1.0), (-2.5, 0.5), (10.0, 3.0)), x in (-3.0, -0.4, 0.0, 1.7, 8.0)
        d = Normal(μ, σ)
        @test logdensityof(d, x) ≈ Distributions.logpdf(ref(μ, σ), x)
        @test cdf(d, x) ≈ Distributions.cdf(ref(μ, σ), x)
        @test ccdf(d, x) ≈ Distributions.ccdf(ref(μ, σ), x)
        @test logcdf(d, x) ≈ Distributions.logcdf(ref(μ, σ), x)
        @test logccdf(d, x) ≈ Distributions.logccdf(ref(μ, σ), x)
    end
    for (μ, σ) in ((0.0, 1.0), (-2.5, 0.5)), p in (0.01, 0.25, 0.5, 0.9, 0.999)
        @test quantile(Normal(μ, σ), p) ≈ Distributions.quantile(ref(μ, σ), p)
    end
    for (μ, σ) in ((0.0, 1.0), (-2.5, 0.5))
        d, r = Normal(μ, σ), ref(μ, σ)
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "sampling" begin
    d = Normal(1.5, 2.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Normal(0.0f0, 1.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(isfinite, v)

    # For `x = μ + σz`, the derivatives are 1 and `z`.
    dμ = ForwardDiff.derivative(m -> rand(Xoshiro(7), Normal(m, 2.0)), 1.5)
    @test dμ == 1.0
    z = (rand(Xoshiro(7), Normal(0.0, 1.0)))
    dσ = ForwardDiff.derivative(s -> rand(Xoshiro(7), Normal(1.5, s)), 2.0)
    @test dσ ≈ z
end
