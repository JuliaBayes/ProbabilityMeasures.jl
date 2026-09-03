using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using LinearAlgebra: I, LowerTriangular
using Random: Random, Xoshiro
using SpecialFunctions: digamma, loggamma
using Test

# A lower-triangular factor whose scale matrix has distinct, correlated entries.
function factor(p, T=Float64)
    return T[i == j ? 1 + i//2 : (i > j ? (i - j)//4 : 0) for i in 1:p, j in 1:p]
end

@testset "conformance" begin
    for d in (
        Wishart(5.0, [1.0 0.0; 0.0 1.0]),
        Wishart(3.5, [2.0 0.0; 0.5 1.5]),
        Wishart(4.0f0, Float32[1.0 0.0; -0.25 0.5]),
        Wishart(7.0, factor(3)),
    )
        test_measure(d; name=string(d))
    end
end

@testset "traits" begin
    d = Wishart(5.0, [2.0 0.0; 0.5 1.5])
    @test d isa AbstractProbabilityMeasure{Matrixvariate,Continuous}
    @test d isa ContinuousMatrixvariateMeasure
    @test !(d isa ContinuousMultivariateMeasure)
    @test string(d) == "Wishart(ν=5.0, L=[2.0 0.0; 0.5 1.5])"
    @test keys(params(d)) === (:ν, :L)
    @test params(d) == (ν=5.0, L=[2.0 0.0; 0.5 1.5])
end

@testset "no promotion at construction" begin
    L = [2.0 0.0; 0.5 1.5]
    @test typeof(Wishart(5, L)) === Wishart{Int,Matrix{Float64}}
    @test typeof(Wishart(5.0f0, Float32.(L))) === Wishart{Float32,Matrix{Float32}}

    @test eltype(Wishart(5, [2 0; 1 2])) === Matrix{Float64}
    @test eltype(Wishart(5.0f0, Float32.(L))) === Matrix{Float32}
end

@testset "precision follows the argument, not the parameters" begin
    exact = Wishart(5, [2 0; 1 2])
    X = [6.0 1.0; 1.0 4.0]
    @test logdensityof(exact, Float32.(X)) isa Float32
    @test logdensityof(exact, big.(X)) isa BigFloat

    # Integer parameters must not reduce `BigFloat` precision.
    widened = logdensityof(Wishart(big"5.0", big.([2 0; 1 2])), big.(X))
    @test abs(logdensityof(exact, big.(X)) - widened) < 1e-70
end

@testset "construction never validates" begin
    p = 2
    identity2 = Matrix{Float64}(I, p, p)
    # A measure at `p - 1` degrees of freedom is singular and has no density.
    for d in (
        Wishart(1.0, identity2),
        Wishart(0.5, identity2),
        Wishart(Inf, identity2),
        Wishart(5.0, [0.0 0.0; 0.5 1.5]),
        Wishart(5.0, [-1.0 0.0; 0.5 1.5]),
        Wishart(5.0, [NaN 0.0; 0.5 1.5]),
    )
        @test !checkparams(d)
        @test !isfinite(logdensityof(d, [6.0 1.0; 1.0 4.0]))
    end
    @test checkparams(Wishart(1.0001, identity2))
    @test !checkparams(Wishart(5.0, [1.0 0.0 0.0; 0.0 1.0 0.0]))

    @test_throws DomainError validateparams(Wishart(1.0, identity2))
    @test validateparams(Wishart(5.0, identity2)) isa Wishart
end

@testset "support" begin
    d = Wishart(5.0, [2.0 0.0; 0.5 1.5])
    @test support(d) === PositiveDefiniteMatrices(2)

    @test insupport(d, [1.0 0.0; 0.0 1.0])
    @test insupport(d, [4.0 1.0; 1.0 2.5])
    # Singular, indefinite, asymmetric, mis-shaped, and non-finite all fall outside.
    @test !insupport(d, [1.0 1.0; 1.0 1.0])
    @test !insupport(d, [-1.0 0.0; 0.0 1.0])
    @test !insupport(d, [1.0 0.5; 0.4 1.0])
    @test !insupport(d, [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    @test !insupport(d, [NaN 0.0; 0.0 1.0])

    # Draws land exactly in the support, not a rounding error away from it.
    for k in 1:20
        @test insupport(d, rand(Xoshiro(k), d))
    end
end

@testset "density is total off the support" begin
    d = Wishart(5.0, [2.0 0.0; 0.5 1.5])
    for X in (
        fill(Inf, 2, 2),
        fill(-Inf, 2, 2),
        fill(NaN, 2, 2),
        fill(floatmax(Float64), 2, 2),
        zeros(2, 2),
        -ones(2, 2),
        zeros(3, 3),
        zeros(0, 0),
    )
        @test !isfinite(logdensityof(d, X))
    end
end

@testset "one dimension is a scaled chi-squared" begin
    # `Wishart(ν, σ)` on 1-by-1 matrices is `Gamma(ν/2, 2σ²)`.
    for ν in (1.5, 3.0, 8.0), σ in (0.5, 1.0, 2.0), x in (0.3, 1.0, 7.0)
        d, g = Wishart(ν, fill(σ, 1, 1)), Gamma(ν / 2, 2σ^2)
        @test logdensityof(d, fill(x, 1, 1)) ≈ logdensityof(g, x)
        @test mean(d)[1] ≈ mean(g)
        @test var(d)[1] ≈ var(g)
        @test entropy(d) ≈ entropy(g)
    end
end

@testset "reference numerics against Distributions.jl" begin
    for p in 1:4, ν in (p - 0.5, float(p), p + 0.5, p + 3.0, 20.0)
        ν > p - 1 || continue
        L = factor(p)
        d, r = Wishart(ν, L), Distributions.Wishart(ν, Matrix(L * L'))
        for k in 1:5
            X = rand(Xoshiro(k), d)
            @test logdensityof(d, X) ≈ Distributions.logpdf(r, X)
            @test densityof(d, X) ≈ Distributions.pdf(r, X)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test cov(d) ≈ Distributions.cov(r)
        @test std(d) ≈ sqrt.(Distributions.var(r))
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "the multivariate gamma function matches its product" begin
    for p in 1:4, a in (p / 2 + 0.1, p / 2 + 1, 5.0, 12.5)
        expected = (p * (p - 1) / 4) * log(π)
        expected += sum(j -> loggamma(a + (1 - j) / 2), 1:p)
        @test ProbabilityMeasures.logmvgamma(p, a) ≈ expected
        @test ProbabilityMeasures.mvdigamma(p, a) ≈ sum(j -> digamma(a + (1 - j) / 2), 1:p)
    end
    # Below the domain of the product, both report `NaN` rather than throwing.
    @test isnan(ProbabilityMeasures.logmvgamma(3, 1.0))
    @test isnan(ProbabilityMeasures.mvdigamma(3, 1.0))
end

@testset "Cholesky factorization is exact and total" begin
    A = [4.0 1.0 0.5; 1.0 3.0 -0.25; 0.5 -0.25 2.0]
    C = ProbabilityMeasures.cholfactor(A)
    @test C ≈ LowerTriangular(C)
    @test C * C' ≈ A
    # Forward substitution inverts the factor.
    b = [1.0, -2.0, 0.5]
    @test C * ProbabilityMeasures.forwardsolve(C, b) ≈ b

    # An indefinite matrix gives a non-finite factor instead of an error.
    @test any(isnan, ProbabilityMeasures.cholfactor([1.0 2.0; 2.0 1.0]))
    @test any(!isfinite, ProbabilityMeasures.cholfactor(zeros(2, 2)))
end

@testset "sampling" begin
    d = Wishart(5.0, [2.0 0.0; 0.5 1.5])
    @test rand(Xoshiro(1), d) isa Matrix{Float64}
    @test rand(Xoshiro(1), Wishart(5.0f0, Float32[2.0 0.0; 0.5 1.5])) isa Matrix{Float32}
    @test size(rand(Xoshiro(1), d)) == (2, 2)
    @test length(rand(Xoshiro(1), d, 3)) == 3
    @test eltype(rand(Xoshiro(1), d, 3)) === Matrix{Float64}

    # Degrees of freedom just above the singular bound take Gamma's boosted sampler.
    for (ν, p) in ((1.2, 2), (5.0, 2), (3.1, 3), (12.0, 3))
        m = Wishart(ν, factor(p))
        draws = [rand(Xoshiro(k), m) for k in 1:100_000]
        average = sum(draws) / length(draws)
        spread = sum(X -> (X .- average) .^ 2, draws) / (length(draws) - 1)
        @test all(X -> insupport(m, X), draws[1:100])
        @test average ≈ mean(m) rtol = 0.03
        @test spread ≈ var(m) rtol = 0.10
    end
end

@testset "log-density gradient with respect to the parameters" begin
    L = [2.0 0.0; 0.5 1.5]
    X = [6.0 1.0; 1.0 4.0]
    p0 = [5.0; vec(L)]
    rebuild(v) = Wishart(v[1], reshape(v[2:end], 2, 2))
    g = ForwardDiff.gradient(v -> logdensityof(rebuild(v), X), p0)

    h = 1e-6
    for i in eachindex(p0)
        step = [j == i ? h : 0.0 for j in eachindex(p0)]
        expected =
            (logdensityof(rebuild(p0 .+ step), X) - logdensityof(rebuild(p0 .- step), X)) /
            2h
        @test g[i] ≈ expected rtol = 1e-5 atol = 1e-7
    end
    # The upper triangle of the factor is never read.
    @test iszero(g[4])
end

@testset "sample derivative follows the parameters" begin
    L = [2.0 0.0; 0.5 1.5]
    p0 = [5.0; vec(L)]
    rebuild(v) = Wishart(v[1], reshape(v[2:end], 2, 2))
    draw(v) = sum(rand(Xoshiro(7), rebuild(v)))
    g = ForwardDiff.gradient(draw, p0)

    h = 1e-6
    for i in eachindex(p0)
        step = [j == i ? h : 0.0 for j in eachindex(p0)]
        @test g[i] ≈ (draw(p0 .+ step) - draw(p0 .- step)) / 2h rtol = 1e-4 atol = 1e-6
    end
    @test !iszero(g[1])
end
