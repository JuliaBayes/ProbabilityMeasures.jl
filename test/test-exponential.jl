using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    # Widen `θ` because Distributions.jl computes its rate at the parameter's precision.
    reference_logpdf(m, x) =
        Distributions.logpdf(Distributions.Exponential(Float64(m.θ)), x)
    for d in (Exponential(1.0), Exponential(2.5), Exponential(3.0f0), Exponential(2))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(1.0, 1.0)
    @test typeof(Exponential(dual)) === Exponential{typeof(dual)}
    @test typeof(Exponential(1)) === Exponential{Int}
    @test typeof(Exponential(1.0f0)) === Exponential{Float32}

    @test Exponential(1.0f0).θ isa Float32
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Exponential(2), 1.0f0) isa Float32
    @test logdensityof(Exponential(2), big"1.0") isa BigFloat

    # Integer parameters must not reduce `BigFloat` precision.
    exact = logdensityof(Exponential(2), big"1.0")
    full = logdensityof(Exponential(big"2.0"), big"1.0")
    @test abs(exact - full) < 1e-70

    # Exact rational inputs must still return floating-point values.
    @test logdensityof(Exponential(2), 1//2) ≈ -0.25 - log(2.0)
    @test logdensityof(Exponential(2//1), 1//2) isa Float64
    @test logdensityof(Exponential(2), -1//2) === -Inf
    @test logdensityof(Exponential(2//1), -1//2) === -Inf
    @test logcdf(Exponential(2//1), -1//2) === -Inf
end

@testset "construction never validates" begin
    d = Exponential(-1.0)
    @test !checkparams(d)
    @test isnan(logdensityof(d, 1.0))
    @test checkparams(Exponential(1.0))
    @test !checkparams(Exponential(Inf))
end

@testset "logccdf beats log(ccdf) in the tail" begin
    d = Exponential(1.0)
    # The CCDF underflows here, but its logarithm should remain finite.
    @test ccdf(d, 1000.0) == 0.0
    @test isfinite(logccdf(d, 1000.0))
    @test logccdf(d, 1000.0) ≈ Distributions.logccdf(Distributions.Exponential(), 1000.0)
end

@testset "reference numerics against Distributions.jl" begin
    ref(θ) = Distributions.Exponential(θ)
    for θ in (0.4, 1.0, 3.0), x in (0.0, 0.2, 1.7, 8.0)
        d = Exponential(θ)
        @test logdensityof(d, x) ≈ Distributions.logpdf(ref(θ), x)
        @test cdf(d, x) ≈ Distributions.cdf(ref(θ), x)
        @test ccdf(d, x) ≈ Distributions.ccdf(ref(θ), x)
        @test logcdf(d, x) ≈ Distributions.logcdf(ref(θ), x)
        @test logccdf(d, x) ≈ Distributions.logccdf(ref(θ), x)
    end
    for θ in (0.4, 1.0, 3.0), p in (0.01, 0.25, 0.5, 0.9, 0.999)
        @test quantile(Exponential(θ), p) ≈ Distributions.quantile(ref(θ), p)
    end
    for θ in (0.4, 1.0, 3.0)
        d, r = Exponential(θ), ref(θ)
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "sampling" begin
    d = Exponential(1.5)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Exponential(1.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(isfinite, v)

    # For `x = θe`, the derivative is `e = x/θ`.
    x = rand(Xoshiro(7), Exponential(1.5))
    dθ = ForwardDiff.derivative(θ -> rand(Xoshiro(7), Exponential(θ)), 1.5)
    @test dθ ≈ x / 1.5
end
