using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

reference(m) = Distributions.Multinomial(Int(m.n), Float64.(m.p))
reference_logpdf(m, x) = Distributions.logpdf(reference(m), Int.(x))

"Every vector of `k` non-negative integers that sums to `n`."
function compositions(n::Int, k::Int)
    k == 1 && return [[n]]
    return [[i; rest] for i in 0:n for rest in compositions(n - i, k - 1)]
end

@testset "conformance" begin
    measures = (
        Multinomial(4, [0.2, 0.3, 0.5]),
        Multinomial(1, [0.7, 0.3]),
        Multinomial(6, Float32[0.25, 0.75]),
        Multinomial(3, [1.0]),
        Multinomial(5, [0.1, 0.2, 0.3, 0.4]),
    )
    for d in measures
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Multinomial(4, [0.2, 0.3, 0.5])
    @test d isa AbstractProbabilityMeasure{Multivariate,Discrete}
    @test d isa DiscreteMultivariateMeasure
    @test !(d isa ContinuousMultivariateMeasure)
    @test !(d isa DiscreteUnivariateMeasure)
    @test string(d) == "Multinomial(n=4, p=[0.2, 0.3, 0.5])"
    @test params(d).n === 4
    @test params(d).p === d.p
end

@testset "no promotion at construction" begin
    # `n` stays an integer because it sets the support and the sampling loop length.
    @test typeof(Multinomial(4, [0.5, 0.5])) === Multinomial{Int,Vector{Float64}}
    @test typeof(Multinomial(Int32(4), Float32[0.5, 0.5])) ===
        Multinomial{Int32,Vector{Float32}}
    @test typeof(Multinomial(4, [1//2, 1//2])) === Multinomial{Int,Vector{Rational{Int}}}

    # Counts use the probabilities' floating-point type.
    @test eltype(Multinomial(4, [0.5, 0.5])) === Vector{Float64}
    @test eltype(Multinomial(4, Float32[0.5, 0.5])) === Vector{Float32}
    @test eltype(Multinomial(4, [1//2, 1//2])) === Vector{Float64}
end

@testset "precision follows the argument, not the parameters" begin
    d = Multinomial(3, [1//2, 1//2])
    @test logdensityof(d, Float32[2, 1]) isa Float32
    @test logdensityof(d, [big"2.0", big"1.0"]) isa BigFloat
    @test logdensityof(Multinomial(3, Float32[0.5, 0.5]), [2.0, 1.0]) isa Float64

    # The coefficient must use `BigFloat` even though the counts are integers.
    exact = logdensityof(d, [big"2.0", big"1.0"])
    @test abs(exact - (log(big"3.0") - 3 * log(big"2.0"))) < 1e-70
end

@testset "construction never validates" begin
    negative = Multinomial(-1, [0.5, 0.5])
    @test !checkparams(negative)
    @test logdensityof(negative, [1.0, 1.0]) == -Inf

    # An unnormalized `p` still gives a finite log-density.
    unnormalized = Multinomial(2, [0.5, 0.5, 0.5])
    @test !checkparams(unnormalized)
    @test isfinite(logdensityof(unnormalized, [1.0, 1.0, 0.0]))

    below = Multinomial(2, [-0.5, 1.5])
    @test !checkparams(below)
    @test isnan(logdensityof(below, [1.0, 1.0]))
    # A negative probability drops out where its count is zero.
    @test isfinite(logdensityof(below, [0.0, 2.0]))

    @test !checkparams(Multinomial(2, [NaN, 0.5, 0.5]))
    @test !checkparams(Multinomial(2, Float64[]))
    @test checkparams(Multinomial(0, [0.2, 0.8]))

    @test_throws DomainError validateparams(Multinomial(2, [0.5, 0.5, 0.5]))
    valid = Multinomial(2, [0.5, 0.5])
    @test validateparams(valid) === valid
end

@testset "support" begin
    d = Multinomial(4, [0.2, 0.3, 0.5])
    @test support(d) === CountVectors(4, 3)
    @test isbits(support(d))

    @test insupport(d, [2.0, 1.0, 1.0])
    @test insupport(d, [4.0, 0.0, 0.0])
    @test insupport(d, [2, 1, 1])
    @test !insupport(d, [2.0, 1.0])
    @test !insupport(d, [2.0, 1.0, 1.0, 0.0])
    @test !insupport(d, [1.0, 1.0, 1.0])
    @test !insupport(d, [5.0, 0.0, -1.0])
    @test !insupport(d, [2.5, 0.5, 1.0])
    @test !insupport(d, [NaN, 1.0, 1.0])
    @test !insupport(d, [Inf, -Inf, 1.0])

    # No draws leaves one count vector; a negative count of draws leaves none.
    @test insupport(Multinomial(0, [0.5, 0.5]), [0.0, 0.0])
    @test !insupport(Multinomial(-1, [0.5, 0.5]), [0.0, 0.0])
end

@testset "density is total off the support" begin
    d = Multinomial(4, [0.2, 0.3, 0.5])
    outside = (
        [1.0, 1.0, 1.0],
        [2.5, 0.5, 1.0],
        [5.0, 0.0, -1.0],
        [Inf, 0.0, 0.0],
        [-Inf, 0.0, 0.0],
        [NaN, 0.0, 0.0],
        fill(floatmax(Float64), 3),
        [4.0, 0.0],
        Float64[],
    )
    for x in outside
        @test logdensityof(d, x) == -Inf
    end
    for x in compositions(4, 3)
        @test isfinite(logdensityof(d, float.(x)))
    end
end

@testset "normalization over the support" begin
    for d in (
        Multinomial(4, [0.2, 0.3, 0.5]),
        Multinomial(6, [0.5, 0.5]),
        Multinomial(3, [0.1, 0.2, 0.3, 0.4]),
        Multinomial(0, [0.2, 0.8]),
    )
        n, k = Int(d.n), length(d.p)
        @test sum(x -> densityof(d, float.(x)), compositions(n, k)) ≈ 1
    end
end

@testset "one draw is a one-hot Categorical" begin
    p = [0.2, 0.3, 0.5]
    d, c = Multinomial(1, p), Categorical(p)
    for i in eachindex(p)
        x = [float(j == i) for j in eachindex(p)]
        @test logdensityof(d, x) ≈ logdensityof(c, float(i))
    end
    @test mean(d) ≈ p
end

@testset "two categories are a Binomial in the first count" begin
    for (n, p) in ((1, 0.5), (5, 0.3), (10, 0.75))
        d, b = Multinomial(n, [p, 1 - p]), Binomial(n, p)
        for k in 0:n
            @test logdensityof(d, [float(k), float(n - k)]) ≈ logdensityof(b, float(k))
        end
        @test mean(d)[1] ≈ mean(b)
        @test var(d)[1] ≈ var(b)
    end
end

@testset "reference numerics against Distributions.jl" begin
    measures = (
        Multinomial(4, [0.2, 0.3, 0.5]),
        Multinomial(1, [0.7, 0.3]),
        Multinomial(10, [0.1, 0.2, 0.3, 0.4]),
        Multinomial(7, [0.5, 0.5]),
    )
    for d in measures
        r = reference(d)
        for x in compositions(Int(d.n), length(d.p))
            @test logdensityof(d, float.(x)) ≈ Distributions.logpdf(r, x)
            @test densityof(d, float.(x)) ≈ Distributions.pdf(r, x)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test cov(d) ≈ Distributions.cov(r)
    end
end

@testset "moments" begin
    d = Multinomial(6, [0.2, 0.3, 0.5])
    @test mean(d) ≈ [1.2, 1.8, 3.0]
    @test var(d) ≈ [6 * 0.2 * 0.8, 6 * 0.3 * 0.7, 6 * 0.5 * 0.5]
    @test var(d) ≈ [cov(d)[i, i] for i in axes(cov(d), 1)]
    @test cov(d) ≈ transpose(cov(d))
    # Counts sum to `n`, so the off-diagonal entries are negative.
    @test cov(d)[1, 2] ≈ -6 * 0.2 * 0.3
    @test all(cov(d)[i, j] <= 0 for i in 1:3, j in 1:3 if i != j)
    @test std(d) ≈ sqrt.(var(d))
    # Each row of the covariance sums to zero, since the total is fixed.
    @test all(≈(0; atol=1e-12), sum(cov(d); dims=2))

    nsamples = 200_000
    draws = rand(Xoshiro(20250801), d, nsamples)
    samples = reduce(hcat, draws)
    m = vec(sum(samples; dims=2)) ./ nsamples
    # Allow five standard errors for each sampled mean.
    @test m ≈ mean(d) atol = 5 * maximum(std(d)) / sqrt(nsamples)
    centered = samples .- m
    @test centered * centered' ./ (nsamples - 1) ≈ cov(d) rtol = 20 / sqrt(nsamples)
end

@testset "degenerate parameters" begin
    # No draws puts all the mass on the zero count vector.
    d = Multinomial(0, [0.2, 0.8])
    @test checkparams(d)
    @test logdensityof(d, [0.0, 0.0]) == 0.0
    @test logdensityof(d, [1.0, 0.0]) == -Inf
    @test mean(d) == [0.0, 0.0]
    @test var(d) == [0.0, 0.0]
    @test all(==([0.0, 0.0]), rand(Xoshiro(1), d, 8))

    # A zero count must win over the infinite log of a zero probability.
    z = Multinomial(3, [0.0, 1.0])
    @test logdensityof(z, [0.0, 3.0]) == 0.0
    @test logdensityof(z, [1.0, 2.0]) == -Inf
    @test all(==([0.0, 3.0]), rand(Xoshiro(1), z, 8))

    # One category collects every draw.
    single = Multinomial(3, [1.0])
    @test logdensityof(single, [3.0]) == 0.0
    @test rand(Xoshiro(1), single) == [3.0]
end

@testset "log-density gradient with respect to p" begin
    p = [0.2, 0.3, 0.5]
    for x in compositions(4, 3)
        g = ForwardDiff.gradient(q -> logdensityof(Multinomial(4, q), float.(x)), p)
        @test g ≈ x ./ p
    end
end

@testset "sample derivative is zero" begin
    # Counts change in steps as `p` changes.
    g = ForwardDiff.jacobian(q -> rand(Xoshiro(7), Multinomial(5, q)), [0.2, 0.3, 0.5])
    @test all(iszero, g)
end

@testset "sampling" begin
    p = [0.2, 0.3, 0.5]
    d = Multinomial(6, p)
    @test rand(Xoshiro(1), d) isa Vector{Float64}
    @test rand(Xoshiro(1), Multinomial(6, Float32[0.25, 0.75])) isa Vector{Float32}
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Vector{Float64}

    draws = rand(Xoshiro(20250801), d, 1000)
    @test all(x -> insupport(d, x), draws)
    @test all(x -> sum(x) == 6, draws)

    v = Vector{Vector{Float64}}(undef, 4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    # Marginal counts follow the category probabilities.
    samples = reduce(hcat, draws)
    @test vec(sum(samples; dims=2)) ./ (1000 * 6) ≈ p atol = 0.02
end
