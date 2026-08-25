using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    function reference_logpdf(m, x)
        return Distributions.logpdf(Distributions.Poisson(m.λ), Int(x))
    end
    for d in (Poisson(0.5), Poisson(4.0), Poisson(2.5f0), Poisson(11.0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Poisson(4.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "Poisson(λ=4.0)"
    @test params(d) === (λ=4.0,)
end

@testset "numeric types" begin
    @test typeof(Poisson(4.0)) === Poisson{Float64}
    @test typeof(Poisson(4.0f0)) === Poisson{Float32}
    @test typeof(Poisson(2)) === Poisson{Int}
    @test typeof(Poisson(5//2)) === Poisson{Rational{Int}}

    @test eltype(Poisson(4.0f0)) === Float32
    @test eltype(Poisson(2)) === Float64
    @test isbits(Poisson(4.0))
end

@testset "precision follows the argument, not the parameter" begin
    @test logdensityof(Poisson(2), 1.0f0) isa Float32
    @test logdensityof(Poisson(2), big"1.0") isa BigFloat

    # An integer rate must preserve a `BigFloat` argument.
    exact = logdensityof(Poisson(2), big"1.0")
    @test abs(exact - (log(big"2.0") - 2)) < 1e-70
end

@testset "construction never validates" begin
    # A negative rate is invalid even though its density is finite at zero.
    negative = Poisson(-1.0)
    @test !checkparams(negative)
    @test isfinite(logdensityof(negative, 0.0))
    @test isnan(logdensityof(negative, 1.0))

    @test !checkparams(Poisson(NaN))
    @test !checkparams(Poisson(Inf))
    @test !isfinite(logdensityof(Poisson(NaN), 0.0))
    @test !isfinite(logdensityof(Poisson(Inf), 0.0))

    @test_throws DomainError validateparams(Poisson(-1.0))
    @test validateparams(Poisson(4.0)) === Poisson(4.0)
end

@testset "support" begin
    d = Poisson(4.0)
    @test support(d) === NonNegativeIntegers()
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === Inf

    @test insupport(d, 0.0)
    @test insupport(d, 7)
    @test insupport(d, 1e6)
    @test !insupport(d, -1.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)
end

@testset "density is total off the support" begin
    d = Poisson(4.0)
    # Off-support values return `-Inf` without throwing.
    for x in (-1.0, -1.5, 2.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for k in 0:20
        @test isfinite(logdensityof(d, float(k)))
    end
end

@testset "a zero rate puts all the mass on zero" begin
    d = Poisson(0.0)
    @test checkparams(d)
    # The `0^0` factor contributes one.
    @test logdensityof(d, 0.0) == 0.0
    @test logdensityof(d, 1.0) == -Inf
    @test mean(d) == 0.0
    @test var(d) == 0.0
    @test entropy(d) == 0.0
    @test cdf(d, 0.0) == 1.0
    @test ccdf(d, 0.0) == 0.0
    @test all(==(0.0), rand(Xoshiro(1), d, 8))
end

@testset "normalization" begin
    # `k <= 400` captures the tail for all tested rates.
    for λ in (0.5, 1.0, 4.0, 11.0, 40.0)
        @test sum(k -> densityof(Poisson(λ), float(k)), 0:400) ≈ 1
    end
end

@testset "reference numerics against Distributions.jl" begin
    for λ in (0.1, 0.5, 1.0, 4.0, 11.0, 40.0)
        d, r = Poisson(λ), Distributions.Poisson(λ)
        for k in 0:Distributions.quantile(r, 0.9999)
            x = float(k)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test densityof(d, x) ≈ Distributions.pdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            # Use absolute tolerance near `log(1) == 0`.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k) atol = 1e-12
        end
        for q in (0.01, 0.25, 0.5, 0.9, 0.999)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the atoms" begin
    d = Poisson(4.0)
    # The CDF is constant between integer outcomes.
    @test cdf(d, 2.0) == cdf(d, 2.999)
    @test cdf(d, -0.5) == 0.0
    # The direct tail sum may round below one.
    @test ccdf(d, -0.5) ≈ 1.0
    @test logcdf(d, -1.0) == -Inf

    # Directly summing both tails avoids cancellation.
    for k in 0:20
        @test cdf(d, float(k)) + ccdf(d, float(k)) ≈ 1.0
    end

    # CDF inversion recovers atoms until the CDF rounds to one.
    ks = [k for k in 0:20 if cdf(d, float(k)) < 1.0]
    @test [quantile(d, cdf(d, float(k))) for k in ks] == float.(ks)

    # Invalid probabilities still map to the support.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "sums stop above the mean" begin
    d = Poisson(4.0)
    # The horizon exhausts the `Float64` tail mass.
    @test cdf(d, float(ProbabilityMeasures.horizon(d))) ≈ 1.0
    # The closed form reports the mass past the horizon, which a sum truncates away.
    @test 0 < ccdf(d, float(ProbabilityMeasures.horizon(d))) < 1e-100
    # The horizon stands in for the infinite quantile at `q = 1`.
    @test quantile(d, 1.0) == float(ProbabilityMeasures.horizon(d))

    # Non-finite and sufficiently negative rates have no usable horizon.
    @test ProbabilityMeasures.horizon(Poisson(NaN)) == 0
    @test ProbabilityMeasures.horizon(Poisson(Inf)) == 0
    @test ProbabilityMeasures.horizon(Poisson(-Inf)) == 0
    @test ProbabilityMeasures.horizon(Poisson(-1e300)) == 0

    # Finite rates can also overflow the `Int` horizon.
    for λ in (1e19, 1e20, 1e300, prevfloat(floatmax(Float64)))
        @test ProbabilityMeasures.horizon(Poisson(λ)) == 0
    end
    # Pin the `Float64` cutoff near `typemax(Int)`.
    @test ProbabilityMeasures.horizon(Poisson(9.2e18)) > 0
    @test ProbabilityMeasures.horizon(Poisson(9.3e18)) == 0
end

@testset "an empty horizon gives NaN, not a partial sum" begin
    for λ in (1e19, 1e300, -100.0)
        d = Poisson(λ)
        @test ProbabilityMeasures.horizon(d) == 0
        @test isnan(cdf(d, 3.0))
        @test isnan(ccdf(d, 3.0))
        @test isnan(quantile(d, 0.5))
        @test isnan(entropy(d))
        @test isnan(rand(Xoshiro(1), d))
    end
end

#=
  A `Dual` is `Real` but not `AbstractFloat`, so it takes the summing path while a plain
  float takes the closed form. Their values have to agree.
=#
@testset "both paths agree" begin
    for λ in (0.5, 1.0, 4.0, 11.0, 40.0, 200.0)
        d = Poisson(λ)
        for k in 0:ceil(Int, λ + 4 * sqrt(λ))
            x = float(k)
            summed = ForwardDiff.value(cdf(d, ForwardDiff.Dual(x, 1.0)))
            @test summed ≈ cdf(d, x) atol = 1e-12
            summed = ForwardDiff.value(ccdf(d, ForwardDiff.Dual(x, 1.0)))
            @test summed ≈ ccdf(d, x) atol = 1e-12
        end
        for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
            summed = ForwardDiff.value(quantile(d, ForwardDiff.Dual(q, 1.0)))
            @test summed == quantile(d, q)
        end
    end
end

@testset "the closed form keeps its type" begin
    for T in (Float32, Float64, BigFloat)
        d = Poisson(T(4))
        @test cdf(d, T(2)) isa T
        @test ccdf(d, T(2)) isa T
        @test quantile(d, T(1) / 2) isa T
    end
    # A `BigFloat` argument reaches the closed form and keeps its precision.
    @test cdf(Poisson(big"4.0"), big"2.0") isa BigFloat
    @test cdf(Poisson(big"4.0"), big"2.0") ≈ cdf(Poisson(4.0), 2.0) atol = 1e-14
end

@testset "log-density gradient with respect to the rate" begin
    for λ in (0.5, 4.0, 11.0), k in 0:12
        g = ForwardDiff.derivative(l -> logdensityof(Poisson(l), float(k)), λ)
        @test g ≈ k / λ - 1
    end
end

@testset "sample derivative is zero" begin
    # Inverse-CDF samples are piecewise constant in `λ`.
    g = ForwardDiff.derivative(l -> rand(Xoshiro(7), Poisson(l)), 4.0)
    @test iszero(g)
end

@testset "sampling" begin
    d = Poisson(4.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Poisson(4.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 100_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 0.02
    @test var(draws) ≈ var(d) atol = 0.05

    # CDF inversion remains valid when `exp(-λ)` underflows.
    @test all(x -> insupport(d, x), rand(Xoshiro(1), Poisson(1000.0), 8))
end
