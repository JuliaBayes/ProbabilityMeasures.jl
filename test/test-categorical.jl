using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    # Distributions.jl uses integer categories and promoted probabilities.
    reference_logpdf(m, x) =
        Distributions.logpdf(Distributions.Categorical(Float64.(m.p)), Int(x))
    measures = (
        Categorical([0.2, 0.3, 0.5]),
        Categorical([0.7, 0.3]),
        Categorical(Float32[0.25, 0.75]),
        Categorical([1.0]),
    )
    for d in measures
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Categorical([0.5, 0.5])
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "Categorical(p=[0.5, 0.5])"
    @test params(d).p === d.p
end

@testset "no promotion at construction" begin
    @test typeof(Categorical([0.5, 0.5])) === Categorical{Vector{Float64}}
    @test typeof(Categorical(Float32[0.5, 0.5])) === Categorical{Vector{Float32}}
    @test typeof(Categorical(1:1)) === Categorical{UnitRange{Int}}

    @test Categorical(Float32[0.5, 0.5]).p isa Vector{Float32}

    # Category values use the probabilities' floating-point type.
    @test eltype(Categorical([0.5, 0.5])) === Float64
    @test eltype(Categorical(Float32[0.5, 0.5])) === Float32
    @test eltype(Categorical(1:1)) === Float64

    # Inline storage depends on the probability container.
    @test !isbits(Categorical([0.5, 0.5]))
    @test isbits(Categorical(1:1))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Categorical([1]), 1.0f0) isa Float32
    @test logdensityof(Categorical([1]), big"1.0") isa BigFloat
    @test logdensityof(Categorical(Float32[0.25, 0.75]), 1.0) isa Float64

    # Probability calculations must keep `BigFloat` precision.
    third = big"1.0" / 3
    exact = logdensityof(Categorical([third, third, 1 - 2 * third]), big"1.0")
    @test exact isa BigFloat
    @test abs(exact - log(third)) < 1e-70
end

@testset "construction never validates" begin
    d = Categorical([-0.5, 1.5])
    @test !checkparams(d)
    @test isnan(logdensityof(d, 1.0))

    @test !checkparams(Categorical(Float64[]))
    @test logdensityof(Categorical(Float64[]), 1.0) == -Inf

    @test !checkparams(Categorical([NaN, 0.5, 0.5]))

    @test checkparams(Categorical([0.2, 0.3, 0.5]))
    @test checkparams(Categorical(Float32[0.25, 0.75]))
    @test checkparams(Categorical([1.0]))

    # Density evaluation cannot detect a sum different from one.
    unnormalized = Categorical([0.5, 0.5, 0.5])
    @test !checkparams(unnormalized)
    @test isfinite(logdensityof(unnormalized, 1.0))
    @test !checkparams(Categorical([0.5, 0.4]))

    # Allow normal floating-point summation error.
    tenth = fill(0.1, 10)
    @test sum(tenth) != 1.0
    @test checkparams(Categorical(tenth))
    @test Distributions.isprobvec(tenth)
end

@testset "support" begin
    d = Categorical([0.2, 0.3, 0.5])
    @test support(d) === IntegerRange(1, 3)
    @test minimum(support(d)) === 1
    @test maximum(support(d)) === 3

    @test insupport(d, 1.0)
    @test insupport(d, 3.0)
    @test insupport(d, 2)
    @test !insupport(d, 0.0)
    @test !insupport(d, 4.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)

    @test support(Categorical(Float64[])) === IntegerRange(1, 0)
    @test !insupport(Categorical(Float64[]), 1.0)
end

@testset "density is total off the support" begin
    d = Categorical([0.2, 0.3, 0.5])
    for x in (0.0, 4.0, 2.5, -1.0, Inf, -Inf, NaN, floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    @test isfinite(logdensityof(d, 1.0))
    @test isfinite(logdensityof(d, 3.0))
end

@testset "reference numerics against Distributions.jl" begin
    ref(p) = Distributions.Categorical(p)
    vectors = ([0.2, 0.3, 0.5], [0.7, 0.3], [0.1, 0.2, 0.3, 0.4], [1.0])
    for p in vectors
        d, r = Categorical(p), ref(p)
        for i in eachindex(p)
            x = float(i)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, i)
            @test densityof(d, x) ≈ Distributions.pdf(r, i)
            @test cdf(d, x) ≈ Distributions.cdf(r, i)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, i)
            @test logcdf(d, x) ≈ Distributions.logcdf(r, i)
            @test logccdf(d, x) ≈ Distributions.logccdf(r, i)
        end
        for q in (0.0, 0.01, 0.25, 0.5, 0.9, 0.999, 1.0)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test median(d) == Distributions.median(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the categories" begin
    d = Categorical([0.2, 0.3, 0.5])
    # The CDF is constant between categories.
    @test cdf(d, 1.0) == cdf(d, 1.999) == 0.2
    @test ccdf(d, 1.0) == ccdf(d, 1.999)
    @test ccdf(d, 1.0) ≈ 0.8
    @test cdf(d, 0.0) == 0.0
    @test cdf(d, 10.0) == 1.0
    @test logcdf(d, 0.0) == -Inf
    @test logccdf(d, 3.0) == -Inf

    @test [quantile(d, cdf(d, x)) for x in 1.0:3.0] == [1.0, 2.0, 3.0]

    # Out-of-range probabilities still return a category.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "log-density gradient with respect to p" begin
    p = [0.2, 0.3, 0.5]
    # The gradient of `log(p[x])` is nonzero only at `x`.
    for x in eachindex(p)
        g = ForwardDiff.gradient(q -> logdensityof(Categorical(q), float(x)), p)
        expected = zeros(length(p))
        expected[x] = inv(p[x])
        @test g ≈ expected
    end
end

@testset "sample derivative is zero" begin
    # A sample changes in steps as `p` changes.
    g = ForwardDiff.gradient(q -> rand(Xoshiro(7), Categorical(q)), [0.2, 0.3, 0.5])
    @test all(iszero, g)
end

@testset "sampling" begin
    p = [0.2, 0.3, 0.5]
    d = Categorical(p)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Categorical(Float32[0.25, 0.75])) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    @test all(==(1.0), rand(Xoshiro(1), Categorical([1.0]), 16))

    # Sample frequencies should match the category probabilities.
    draws = rand(Xoshiro(20250801), d, 200_000)
    freq = [count(==(float(i)), draws) / length(draws) for i in eachindex(p)]
    @test freq ≈ p atol = 0.005
end
