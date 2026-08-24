using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using LinearAlgebra: diag
using Random: Xoshiro
using Test

reference(d) = Distributions.Multinomial(d.n, Float64.(d.p))

# Count vectors of length `k` that sum to `n`.
counts(n, k) = k == 1 ? [[n]] : [[i; c] for i in 0:n for c in counts(n - i, k - 1)]

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(reference(m), Int.(x))
    for d in (
        Multinomial(5, [0.2, 0.3, 0.5]),
        Multinomial(4, Float32[0.25, 0.25, 0.5]),
        Multinomial(3, [1.0]),
    )
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits and construction" begin
    d = Multinomial(5, [0.2, 0.3, 0.5])
    @test d isa DiscreteMultivariateMeasure
    @test string(d) == "Multinomial(n=5, p=[0.2, 0.3, 0.5])"
    @test params(d) === (n=5, p=d.p)

    @test typeof(Multinomial(Int32(5), Float32[0.25, 0.75])) ===
        Multinomial{Int32,Vector{Float32}}
    @test eltype(d) === Vector{Float64}
    @test eltype(Multinomial(5, Float32[0.25, 0.75])) === Vector{Float32}
end

@testset "parameter validation" begin
    @test checkparams(Multinomial(5, [0.2, 0.3, 0.5]))
    @test checkparams(Multinomial(0, [0.2, 0.3, 0.5]))
    @test checkparams(Multinomial(3, [1.0]))

    for d in (
        Multinomial(-1, [0.5, 0.5]),
        Multinomial(2, Float64[]),
        Multinomial(2, [-0.5, 1.5]),
        Multinomial(2, [0.5, 0.4]),
        Multinomial(2, [0.5, NaN]),
        Multinomial(2, [0.5, Inf]),
    )
        @test !checkparams(d)
        @test_throws DomainError validateparams(d)
    end
end

@testset "support" begin
    s = IntegerSimplex(5, 3)
    @test support(Multinomial(5, [0.2, 0.3, 0.5])) === s
    @test insupport(s, [0, 0, 5])
    @test insupport(s, [1.0, 2.0, 2.0])
    @test !insupport(s, [1, 4])
    @test !insupport(s, [1, 2, 1])
    @test !insupport(s, [-1, 2, 4])
    @test !insupport(s, [1.5, 1.5, 2.0])
    @test !insupport(s, [NaN, 0.0, 5.0])
    @test !insupport(s, [Inf, 0.0, 5.0])
    @test !insupport(s, Float32[1.0, 4.0])
    @test insupport(s, Float32[1.0, 2.0, 2.0])
    @test !insupport(IntegerSimplex(0, 2), [typemax(UInt), one(UInt)])
    @test isbits(s)
end

@testset "density" begin
    d = Multinomial(5, [0.2, 0.3, 0.5])
    r = reference(d)
    @test logdensityof(d, [1.0, 2.0, 2.0]) ≈ Distributions.logpdf(r, [1, 2, 2])

    @test logdensityof(d, [1.0, 4.0]) == -Inf
    @test logdensityof(d, [1.5, 1.5, 2.0]) == -Inf

    @test logdensityof(Multinomial(5, [0.0, 0.0, 1.0]), [0.0, 0.0, 5.0]) == 0.0
    @test logdensityof(Multinomial(5, [0.0, 0.0, 1.0]), [1.0, 0.0, 4.0]) == -Inf
end

@testset "degenerate cases" begin
    zero_trials = Multinomial(0, [0.2, 0.3, 0.5])
    @test logdensityof(zero_trials, [0.0, 0.0, 0.0]) == 0.0
    @test mean(zero_trials) == zeros(3)
    @test cov(zero_trials) == zeros(3, 3)
    @test rand(Xoshiro(0x4d414749), zero_trials) == zeros(3)

    one_category = Multinomial(4, [1.0])
    @test logdensityof(one_category, [4.0]) == 0.0
    @test mean(one_category) == [4.0]
    @test cov(one_category) == zeros(1, 1)
    @test rand(Xoshiro(0x4d495341544f), one_category) == [4.0]
end

@testset "moments and normalization" begin
    d = Multinomial(5, [0.2, 0.3, 0.5])
    r = reference(d)
    @test mean(d) ≈ Distributions.mean(r)
    @test cov(d) ≈ Distributions.cov(r)
    @test var(d) ≈ diag(Distributions.cov(r))

    total = sum(0:5) do x₁
        sum(0:(5 - x₁)) do x₂
            densityof(d, float.([x₁, x₂, 5 - x₁ - x₂]))
        end
    end
    @test total ≈ 1
end

@testset "reference numerics against Distributions.jl" begin
    for d in (Multinomial(5, [0.2, 0.3, 0.5]), Multinomial(4, Float32[0.25, 0.25, 0.5]))
        T, r = eltype(eltype(d)), reference(d)
        for x in counts(d.n, length(d.p))
            value = logdensityof(d, convert.(T, x))
            @test value isa T
            @test value ≈ Distributions.logpdf(r, x) rtol = sqrt(eps(T))
        end
    end

    # The evaluation point can widen the result type.
    @test logdensityof(Multinomial(4, Float32[0.25, 0.25, 0.5]), [2.0, 1.0, 1.0]) isa
        Float64
end

# The conformance suite skips allocation checks for multivariate measures.
@testset "the density does not allocate" begin
    d, x = Multinomial(5, [0.2, 0.3, 0.5]), [1.0, 2.0, 2.0]
    logdensityof(d, x) # compile first
    insupport(d, x)
    @test (@allocated logdensityof(d, x)) == 0
    @test (@allocated insupport(d, x)) == 0
end

@testset "sampling never fills a zero-probability category" begin
    d = Multinomial(6, [0.5, 0.0, 0.5])
    draws = [rand(Xoshiro(seed), d) for seed in 1:500]
    @test all(x -> iszero(x[2]), draws)
    @test all(x -> insupport(d, x), draws)
    @test all(x -> logdensityof(d, x) > -Inf, draws)
end

@testset "sampling" begin
    d, nsamples = Multinomial(5, [0.2, 0.3, 0.5]), 200_000
    @test rand(Xoshiro(0x4144414d), d) isa Vector{Float64}

    draws = rand(Xoshiro(0x4556413031), d, nsamples)
    @test all(x -> insupport(d, x), draws)
    sampled_mean = reduce(+, draws) ./ nsamples
    @test sampled_mean ≈ mean(d) atol = 5 * maximum(std(d)) / sqrt(nsamples)

    centered = [x .- sampled_mean for x in draws]
    sampled_cov = reduce(+, x * x' for x in centered) ./ (nsamples - 1)
    @test sampled_cov ≈ cov(d) rtol = 20 / sqrt(nsamples)
end
