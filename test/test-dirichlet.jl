using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using DifferentiationInterface: AutoForwardDiff
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using LinearAlgebra: diag
using QuadGK: quadgk
using Random: Xoshiro
using Test

reference(d) = Distributions.Dirichlet(Float64.(d.α))

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(reference(m), Float64.(x))
    #=
      A one-entry measure is left out: its only draw is `[1.0]` whatever the shape, so
      the suite's sample-derivative check has nothing to find. It is covered below.
    =#
    #=
      Sampling goes through `Beta`'s quantile, so the same backends apply: Zygote and
      Mooncake differentiate `SpecialFunctions.beta_inc` and lose the derivative with
      respect to every shape but the first.
    =#
    for d in (
        Dirichlet([2.0, 3.0, 5.0]), Dirichlet([1.0, 1.0]), Dirichlet(Float32[0.5, 1.5, 2.0])
    )
        test_measure(
            d;
            name=string(d),
            reference_logpdf=reference_logpdf,
            ad_backends=(AutoForwardDiff(),),
        )
    end
end

@testset "traits and construction" begin
    d = Dirichlet([2.0, 3.0, 5.0])
    @test d isa AbstractProbabilityMeasure{Multivariate,Continuous}
    @test d isa ContinuousMultivariateMeasure
    @test string(d) == "Dirichlet(α=[2.0, 3.0, 5.0])"
    @test params(d) === (α=d.α,)

    @test typeof(Dirichlet(Float32[1.0, 2.0])) === Dirichlet{Vector{Float32}}
    @test eltype(d) === Vector{Float64}
    @test eltype(Dirichlet(Float32[1.0, 2.0])) === Vector{Float32}
    @test eltype(Dirichlet([1, 2])) === Vector{Float64}
end

@testset "parameter validation" begin
    @test checkparams(Dirichlet([2.0, 3.0, 5.0]))
    @test checkparams(Dirichlet([0.1]))

    for bad in (
        Dirichlet(Float64[]),
        Dirichlet([0.0, 1.0]),
        Dirichlet([-1.0, 1.0]),
        Dirichlet([Inf, 1.0]),
        Dirichlet([NaN, 1.0]),
    )
        @test !checkparams(bad)
        @test_throws DomainError validateparams(bad)
    end
end

@testset "support" begin
    s = RealSimplex(3)
    @test support(Dirichlet([2.0, 3.0, 5.0])) === s
    @test isbits(s)

    @test insupport(s, [0.2, 0.3, 0.5])
    @test insupport(s, [0.0, 0.0, 1.0])
    @test insupport(s, Float32[0.25, 0.25, 0.5])
    @test !insupport(s, [0.5, 0.5])
    @test !insupport(s, [0.2, 0.3, 0.6])
    @test !insupport(s, [-0.1, 0.4, 0.7])
    @test !insupport(s, [NaN, 0.5, 0.5])
    @test !insupport(s, [Inf, -Inf, 0.5])
end

@testset "density" begin
    d = Dirichlet([2.0, 3.0, 5.0])
    r = reference(d)
    @test logdensityof(d, [0.2, 0.3, 0.5]) ≈ Distributions.logpdf(r, [0.2, 0.3, 0.5])

    # Wrong length, off the simplex, and negative entries all leave the support.
    @test logdensityof(d, [0.5, 0.5]) == -Inf
    @test logdensityof(d, [0.2, 0.3, 0.6]) == -Inf
    @test logdensityof(d, [-0.1, 0.4, 0.7]) == -Inf

    # A unit shape leaves a zero entry finite rather than producing `0 * -Inf`.
    @test isfinite(logdensityof(Dirichlet([1.0, 1.0, 1.0]), [0.0, 0.5, 0.5]))
    @test logdensityof(Dirichlet([1.0, 1.0, 1.0]), [0.2, 0.3, 0.5]) ≈ log(2.0)
end

@testset "the one-entry measure puts all its mass on one" begin
    d = Dirichlet([4.0])
    @test logdensityof(d, [1.0]) ≈ 0.0
    @test logdensityof(d, [0.5]) == -Inf
    @test mean(d) == [1.0]
    @test cov(d) == zeros(1, 1)
    @test rand(Xoshiro(1), d) == [1.0]
end

@testset "normalization" begin
    #=
      A draw sums to one, so the density is with respect to the first two entries. The
      inner limit follows the outer one, which keeps the point on the simplex.
    =#
    for α in ([2.0, 3.0, 5.0], [1.0, 1.0, 1.0], [0.5, 1.5, 2.0])
        d = Dirichlet(α)
        total, err = quadgk(0.0, 1.0; rtol=1e-9) do x₁
            inner, _ = quadgk(0.0, 1.0 - x₁; rtol=1e-9) do x₂
                densityof(d, [x₁, x₂, 1 - x₁ - x₂])
            end
            return inner
        end
        @test total ≈ 1 atol = max(1e-7, 10err)
    end
end

@testset "reference numerics against Distributions.jl" begin
    for α in ([2.0, 3.0, 5.0], [1.0, 1.0, 1.0], [0.5, 1.5, 2.0], [11.0, 2.0, 7.0])
        d, r = Dirichlet(α), Distributions.Dirichlet(α)
        for x in ([0.2, 0.3, 0.5], [0.25, 0.25, 0.5], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1])
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test cov(d) ≈ Distributions.cov(r)
        @test var(d) ≈ diag(Distributions.cov(r))
        @test std(d) ≈ sqrt.(diag(Distributions.cov(r)))
        @test entropy(d) ≈ Distributions.entropy(r)
    end

    # The evaluation point can widen the result type.
    @test logdensityof(Dirichlet(Float32[1.0, 2.0]), Float32[0.25, 0.75]) isa Float32
    @test logdensityof(Dirichlet(Float32[1.0, 2.0]), [0.25, 0.75]) isa Float64
end

# The conformance suite skips allocation checks for multivariate measures.
@testset "the density does not allocate" begin
    d, x = Dirichlet([2.0, 3.0, 5.0]), [0.2, 0.3, 0.5]
    logdensityof(d, x) # compile first
    insupport(d, x)
    @test (@allocated logdensityof(d, x)) == 0
    @test (@allocated insupport(d, x)) == 0
end

@testset "draws stay on the simplex" begin
    for α in ([2.0, 3.0, 5.0], [0.5, 1.5, 2.0], [1.0, 1.0], [100.0, 0.01])
        d = Dirichlet(α)
        draws = [rand(Xoshiro(seed), d) for seed in 1:200]
        @test all(x -> insupport(d, x), draws)
        @test all(x -> sum(x) ≈ 1, draws)
        @test all(x -> logdensityof(d, x) > -Inf, draws)
    end
end

@testset "sample derivative is not zero" begin
    # Every piece of the stick comes from inverting a continued fraction, so the draw
    # is a smooth function of the shapes.
    g = ForwardDiff.jacobian(p -> rand(Xoshiro(7), Dirichlet(p)), [2.0, 3.0, 5.0])
    @test all(isfinite, g)
    @test any(!iszero, g)
    # The entries sum to one whatever the shapes, so each row of the change sums to zero.
    @test all(≈(0; atol=1e-12), sum(g; dims=1))
end

@testset "sampling" begin
    d, nsamples = Dirichlet([2.0, 3.0, 5.0]), 200_000
    @test rand(Xoshiro(1), d) isa Vector{Float64}
    @test rand(Xoshiro(1), Dirichlet(Float32[1.0, 2.0])) isa Vector{Float32}
    @test length(rand(Xoshiro(1), d, 5)) == 5

    draws = rand(Xoshiro(20250801), d, nsamples)
    sampled_mean = reduce(+, draws) ./ nsamples
    @test sampled_mean ≈ mean(d) atol = 5 * maximum(std(d)) / sqrt(nsamples)

    centered = [x .- sampled_mean for x in draws]
    sampled_cov = reduce(+, x * x' for x in centered) ./ (nsamples - 1)
    @test sampled_cov ≈ cov(d) rtol = 20 / sqrt(nsamples)
end
