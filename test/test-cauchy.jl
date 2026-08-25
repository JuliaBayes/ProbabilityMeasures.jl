using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Cauchy(m.μ, m.σ), x)
    for d in (Cauchy(0.0, 1.0), Cauchy(-2.5, 0.5), Cauchy(3.0f0, 2.0f0))
        test_measure(
            d; name=string(d), reference_logpdf=reference_logpdf, check_moments=false
        )
    end
end

@testset "parameters and undefined moments" begin
    @test typeof(Cauchy(0.0f0, 1)) === Cauchy{Float32,Int}

    d = Cauchy(1.5, 2.0)
    @test isnan(mean(d))
    @test isnan(var(d))
    @test isnan(std(d))
    @test median(d) == 1.5
    @test entropy(d) ≈ Distributions.entropy(Distributions.Cauchy(1.5, 2.0))
end

@testset "stable tails and total quantile" begin
    d = Cauchy()
    xmax = floatmax(Float64)
    @test isfinite(logdensityof(d, xmax))
    @test cdf(d, -xmax) > 0
    @test ccdf(d, xmax) > 0
    @test isfinite(logcdf(d, -xmax))
    @test isfinite(logccdf(d, xmax))
    @test cdf(d, -1.0) == 0.25
    @test ccdf(d, 1.0) == 0.25
    @test quantile(d, 0.25) ≈ -1.0
    @test quantile(d, 0.75) ≈ 1.0

    @test quantile(d, 0.0) == -Inf
    @test quantile(d, -0.0) == -Inf
    @test quantile(d, 1.0) == Inf
    @test isfinite(quantile(d, eps(Float64) / 2))
    @test isfinite(quantile(d, prevfloat(1.0)))
end

@testset "quantile types" begin
    #=
      An out-of-range `p` must widen to the parameters like any other argument. Typing
      the `NaN` by `p` alone drops them and makes the return type a `Union`.
    =#
    @test quantile(Cauchy(0.0, 1.0), 1.5f0) isa Float64
    @test quantile(Cauchy(big(0.0), big(1.0)), 1.5) isa BigFloat
    @test quantile(Cauchy(0.0f0, 1.0f0), 0.25f0) isa Float32
    @test Base.promote_op(quantile, Cauchy{Float64,Float64}, Float32) === Float64
    @test isnan(quantile(Cauchy(0.0, 1.0), 1.5))
    @test isnan(quantile(Cauchy(0.0, 1.0), -0.5))
end
