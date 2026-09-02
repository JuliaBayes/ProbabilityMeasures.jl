using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using SpecialFunctions: digamma, gamma_inc
using Test

@testset "conformance" begin
    # Widen the parameters because Distributions.jl works at their own precision.
    function reference_logpdf(m, x)
        return Distributions.logpdf(Distributions.Gamma(Float64(m.α), Float64(m.θ)), x)
    end
    for d in (Gamma(2.0, 1.0), Gamma(0.5, 3.0), Gamma(4.5f0, 0.5f0), Gamma(3, 2))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Gamma(2.0, 3.0)
    @test d isa AbstractProbabilityMeasure{Univariate,Continuous}
    @test d isa ContinuousUnivariateMeasure
    @test !(d isa DiscreteUnivariateMeasure)
    @test string(d) == "Gamma(α=2.0, θ=3.0)"
    @test params(d) === (α=2.0, θ=3.0)
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(1.0, 1.0)
    @test typeof(Gamma(dual, 1.0)) === Gamma{typeof(dual),Float64}
    @test typeof(Gamma(2, 1.0)) === Gamma{Int,Float64}
    @test typeof(Gamma(2.0f0, 1.0f0)) === Gamma{Float32,Float32}

    # The one-argument form sets the scale to one in the shape's own type.
    @test Gamma(2) === Gamma(2, 1)
    @test Gamma(2.0f0) === Gamma(2.0f0, 1.0f0)

    @test eltype(Gamma(2, 1)) === Float64
    @test eltype(Gamma(2.0f0, 1.0f0)) === Float32
    @test isbits(Gamma(2.0, 3.0))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Gamma(3, 2), 1.0f0) isa Float32
    @test logdensityof(Gamma(3, 2), big"1.0") isa BigFloat

    # Integer parameters must not reduce `BigFloat` precision.
    exact = logdensityof(Gamma(3, 2), big"1.0")
    full = logdensityof(Gamma(big"3.0", big"2.0"), big"1.0")
    @test abs(exact - full) < 1e-70

    # Exact rational inputs must still return floating-point values.
    @test logdensityof(Gamma(3, 2), 1//2) isa Float64
    @test logdensityof(Gamma(3//1, 2//1), 1//2) isa Float64
    @test logdensityof(Gamma(3, 2), -1//2) === -Inf
end

@testset "construction never validates" begin
    for d in (Gamma(-1.0, 1.0), Gamma(0.0, 1.0), Gamma(1.0, -1.0), Gamma(1.0, 0.0))
        @test !checkparams(d)
        @test !isfinite(logdensityof(d, 1.0))
    end
    @test !checkparams(Gamma(Inf, 1.0))
    @test !checkparams(Gamma(1.0, NaN))
    @test checkparams(Gamma(2.0, 3.0))

    @test_throws DomainError validateparams(Gamma(-1.0, 1.0))
    @test validateparams(Gamma(2.0, 3.0)) === Gamma(2.0, 3.0)
end

@testset "invalid parameters give NaN, not a partial answer" begin
    for d in (Gamma(-1.0, 1.0), Gamma(2.0, -1.0), Gamma(NaN, 1.0))
        @test isnan(cdf(d, 1.0))
        @test isnan(ccdf(d, 1.0))
        @test isnan(logcdf(d, 1.0))
        @test isnan(logccdf(d, 1.0))
        @test isnan(quantile(d, 0.5))
        @test isnan(entropy(d))
    end
end

@testset "support" begin
    d = Gamma(2.0, 3.0)
    @test support(d) === PositiveReals()
    @test minimum(support(d)) === 0.0
    @test maximum(support(d)) === Inf

    @test insupport(d, 1e-300)
    @test insupport(d, 1e300)
    @test !insupport(d, 0.0)
    @test !insupport(d, -1.0)
    @test !insupport(d, Inf)
    @test !insupport(d, NaN)
end

@testset "density is total off the support" begin
    # A shape below one has an infinite density at zero, one above it a zero density.
    for d in (Gamma(0.5, 1.0), Gamma(1.0, 1.0), Gamma(2.0, 1.0))
        for x in (0.0, -1.0, -Inf, NaN, -floatmax(Float64))
            @test logdensityof(d, x) == -Inf
        end
        @test !isfinite(logdensityof(d, Inf))
    end
end

@testset "a unit shape is the exponential measure" begin
    for θ in (0.4, 1.0, 3.0), x in (0.2, 1.7, 8.0)
        @test logdensityof(Gamma(1.0, θ), x) ≈ logdensityof(Exponential(θ), x)
        @test cdf(Gamma(1.0, θ), x) ≈ cdf(Exponential(θ), x)
        @test logccdf(Gamma(1.0, θ), x) ≈ logccdf(Exponential(θ), x)
    end
    @test entropy(Gamma(1.0, 2.5)) ≈ entropy(Exponential(2.5))
end

@testset "reference numerics against Distributions.jl" begin
    ref(α, θ) = Distributions.Gamma(α, θ)
    shapes = (0.1, 0.5, 1.0, 2.0, 7.5, 100.0)
    for α in shapes, θ in (0.5, 1.0, 4.0)
        d, r = Gamma(α, θ), ref(α, θ)
        for p in (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
            x = Distributions.quantile(r, p)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
            @test cdf(d, x) ≈ Distributions.cdf(r, x)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, x)
            # Use an absolute tolerance near `log(1) == 0`.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, x) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, x) atol = 1e-12
            @test quantile(d, p) ≈ x rtol = 1e-10
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test median(d) ≈ Distributions.median(r) rtol = 1e-10
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "the regularized incomplete gamma matches SpecialFunctions" begin
    for a in (0.1, 0.5, 1.0, 2.0, 7.5, 100.0, 1000.0), x in (0.01, 0.3, 1.0, 3.0, 20.0)
        y = x * a
        p, q = gamma_inc(a, y)
        # Skip the tails that `gamma_inc` itself rounds to zero.
        p > 0 && @test exp(ProbabilityMeasures.loggammap(a, y)) ≈ p rtol = 1e-11
        q > 0 && @test exp(ProbabilityMeasures.loggammaq(a, y)) ≈ q rtol = 1e-11
    end
end

@testset "log tails stay finite where the probability underflows" begin
    d = Gamma(2.0, 1.0)
    # `Q(2, x) = e^{-x}(1 + x)` and `P(2, x) → x²/2` as `x → 0`.
    @test ccdf(d, 1000.0) == 0.0
    @test logccdf(d, 1000.0) ≈ -1000 + log(1001)
    @test cdf(d, 1e-200) == 0.0
    @test logcdf(d, 1e-200) ≈ 2 * log(1e-200) - log(2)

    # The quantile of a probability this small is representable, and inverts.
    deep = quantile(d, 1e-300)
    @test 0 < deep < 1e-100
    @test logcdf(d, deep) ≈ log(1e-300)
end

@testset "quantile is total and inverts the CDF" begin
    d = Gamma(2.5, 1.5)
    for p in (-0.001, 1.001, -Inf, Inf, NaN)
        @test isnan(quantile(d, p))
    end
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == Inf

    for α in (0.05, 0.5, 1.0, 3.0, 50.0), p in (1e-12, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-9)
        m = Gamma(α, 2.0)
        @test cdf(m, quantile(m, p)) ≈ p rtol = 1e-10
    end
end

@testset "the quantile keeps BigFloat precision" begin
    setprecision(BigFloat, 256) do
        d = Gamma(big"2.5", big"1.5")
        for p in (big"1e-40", big"0.25", big"0.5", big"0.99")
            x = quantile(d, p)
            @test x isa BigFloat
            @test abs(cdf(d, x) - p) < 1e-60 * p
        end
    end
end

@testset "distribution functions keep their type" begin
    for T in (Float32, Float64, BigFloat)
        d = Gamma(T(2), T(3))
        @test cdf(d, T(1)) isa T
        @test ccdf(d, T(1)) isa T
        @test logcdf(d, T(1)) isa T
        @test logccdf(d, T(1)) isa T
        @test quantile(d, T(1) / 4) isa T
        @test entropy(d) isa T
    end
end

@testset "log-density gradient with respect to the parameters" begin
    for α in (0.5, 2.0, 7.5), θ in (0.5, 3.0), x in (0.3, 1.0, 9.0)
        g = ForwardDiff.gradient([α, θ]) do p
            logdensityof(Gamma(p[1], p[2]), x)
        end
        @test g[1] ≈ log(x) - digamma(α) - log(θ)
        @test g[2] ≈ x / θ^2 - α / θ
    end
end

@testset "sampling" begin
    d = Gamma(2.0, 1.5)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Gamma(2.0f0, 1.5f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    # Both the direct method and the boost below a unit shape.
    for α in (0.2, 0.9, 1.0, 3.0, 40.0)
        m = Gamma(α, 2.0)
        draws = rand(Xoshiro(20250801), m, 200_000)
        @test all(x -> insupport(m, x), draws)
        @test mean(draws) ≈ mean(m) atol = 5 * std(m) / sqrt(200_000)
        @test var(draws) ≈ var(m) rtol = 0.05
    end
end

@testset "sample derivative follows the parameters" begin
    # The accept step runs on plain noise, so the draw is a smooth function of both
    # parameters at fixed noise. Compare with a central difference.
    for α in (0.4, 2.0, 9.0), θ in (0.5, 2.0)
        draw(p) = rand(Xoshiro(7), Gamma(p[1], p[2]))
        g = ForwardDiff.gradient(draw, [α, θ])
        h = 1e-6
        dα = (draw([α + h, θ]) - draw([α - h, θ])) / 2h
        dθ = (draw([α, θ + h]) - draw([α, θ - h])) / 2h
        @test g[1] ≈ dα rtol = 1e-4
        @test g[2] ≈ dθ rtol = 1e-6
        @test !iszero(g[1])
    end
end
