using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    # Widen endpoints because Distributions.jl promotes them during validation.
    reference_logpdf(m, x) =
        Distributions.logpdf(Distributions.Uniform(Float64(m.a), Float64(m.b)), x)
    for d in (Uniform(0.0, 1.0), Uniform(-1.0, 2.0), Uniform(0.0f0, 2.0f0), Uniform(0, 1))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(1.0, 1.0)
    @test typeof(Uniform(dual, 2.0)) === Uniform{typeof(dual),Float64}
    @test typeof(Uniform(0, 1)) === Uniform{Int,Int}
    @test typeof(Uniform(0.0f0, 1.0f0)) === Uniform{Float32,Float32}

    @test Uniform(0.0f0, 1).a isa Float32
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Uniform(0, 2), 1.0f0) isa Float32
    @test logdensityof(Uniform(0, 2), big"1.0") isa BigFloat

    # Integer endpoints must not reduce `BigFloat` precision.
    exact = logdensityof(Uniform(0, 2), big"1.0")
    full = logdensityof(Uniform(big"0.0", big"2.0"), big"1.0")
    @test abs(exact - full) < 1e-70

    # Exact rational inputs must still return floating-point values.
    @test logdensityof(Uniform(0, 2), 1//2) === -log(2.0)
    @test logdensityof(Uniform(0//1, 2//1), 1//2) === -log(2.0)
    @test logdensityof(Uniform(0, 2), 7//2) === -Inf
    @test logdensityof(Uniform(0//1, 2//1), -1//2) === -Inf
end

@testset "construction never validates" begin
    d = Uniform(1.0, 0.0)
    @test !checkparams(d)
    @test logdensityof(d, 0.5) == -Inf
    @test checkparams(Uniform(0.0, 1.0))
    @test !checkparams(Uniform(0.0, 0.0))
    @test !checkparams(Uniform(-Inf, 0.0))
end

@testset "support" begin
    d = Uniform(-1.0, 2.0)
    @test support(d) === RealInterval(-1.0, 2.0)
    @test minimum(support(d)) == -1.0
    @test maximum(support(d)) == 2.0

    # Both endpoints belong to the support.
    @test insupport(d, -1.0)
    @test insupport(d, 2.0)
    @test !insupport(d, -1.001)
    @test !insupport(d, NaN)
    @test isfinite(logdensityof(d, -1.0))
    @test isfinite(logdensityof(d, 2.0))
    @test logdensityof(d, nextfloat(2.0)) == -Inf
end

@testset "reference numerics against Distributions.jl" begin
    ref(a, b) = Distributions.Uniform(a, b)
    endpoints = ((0.0, 1.0), (-1.0, 2.0), (2.5, 3.5))
    for (a, b) in endpoints, x in (-2.0, 0.0, 0.4, 2.6, 4.0)
        d = Uniform(a, b)
        @test logdensityof(d, x) ≈ Distributions.logpdf(ref(a, b), x)
        @test cdf(d, x) ≈ Distributions.cdf(ref(a, b), x)
        @test ccdf(d, x) ≈ Distributions.ccdf(ref(a, b), x)
        @test logcdf(d, x) ≈ Distributions.logcdf(ref(a, b), x)
        @test logccdf(d, x) ≈ Distributions.logccdf(ref(a, b), x)
    end
    for (a, b) in endpoints, p in (0.01, 0.25, 0.5, 0.9, 0.999)
        @test quantile(Uniform(a, b), p) ≈ Distributions.quantile(ref(a, b), p)
    end
    for (a, b) in endpoints
        d, r = Uniform(a, b), ref(a, b)
        @test mean(d) ≈ Distributions.mean(r)
        @test median(d) ≈ Distributions.median(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions saturate outside the interval" begin
    d = Uniform(0.0, 1.0)
    @test cdf(d, -1.0) == 0.0
    @test cdf(d, 2.0) == 1.0
    @test ccdf(d, -1.0) == 1.0
    @test ccdf(d, 2.0) == 0.0
    @test logcdf(d, -1.0) == -Inf
    @test logcdf(d, 2.0) == 0.0
    @test logccdf(d, -1.0) == 0.0
    @test logccdf(d, 2.0) == -Inf
end

@testset "sampling" begin
    d = Uniform(-1.0, 2.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Uniform(0.0f0, 1.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    # For `x = a + (b - a)u`, the derivatives are `1 - u` and `u`.
    x = rand(Xoshiro(7), d)
    u = (x - d.a) / (d.b - d.a)
    g = ForwardDiff.gradient(p -> rand(Xoshiro(7), Uniform(p[1], p[2])), [d.a, d.b])
    @test g ≈ [1 - u, u]
end
