using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Geometric(m.p), Int(x))
    for d in (Geometric(0.3), Geometric(0.9), Geometric(0.6f0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Geometric(0.3)
    @test d isa DiscreteUnivariateMeasure
    @test string(d) == "Geometric(p=0.3)"
    @test params(d) === (p=0.3,)
    @test Geometric() === Geometric(0.5)
    @test eltype(Geometric(0.3f0)) === Float32
end

@testset "parameters" begin
    @test checkparams(Geometric(0.3))
    @test checkparams(Geometric(1.0))
    @test !checkparams(Geometric(1.1))
    @test_throws DomainError validateparams(Geometric(0.0))
    @test validateparams(Geometric(0.25)) === Geometric(0.25)
end

@testset "support" begin
    d = Geometric(0.3)
    @test support(d) === NonNegativeIntegers()
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === Inf
    @test insupport(d, 0.0)
    @test insupport(d, 3)
    @test !insupport(d, -1.0)
    @test !insupport(d, 0.5)
    @test !insupport(d, Inf)
    @test !insupport(d, NaN)
end

@testset "reference numerics" begin
    for p in (0.1, 0.3, 0.8, 0.99)
        d, r = Geometric(p), Distributions.Geometric(p)
        for k in 0:50
            x = float(k)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            # Use absolute tolerance near `log(1) == 0`.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k) atol = 1e-12
        end
        for q in (0.0, 0.1, 0.5, 0.99)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test median(d) == Distributions.median(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "normalization" begin
    # `k <= 20000` captures the tail for all tested probabilities.
    for p in (0.1, 0.3, 0.5, 0.9)
        @test sum(k -> densityof(Geometric(p), float(k)), 0:20000) ≈ 1
    end
end

#=
  Inverting the tail overshoots by one whenever `q` lands exactly on a CDF value, so
  the ties are the most bug-prone part of `quantile`. Sweep them, and check minimality
  strictly between the atoms.
=#
@testset "quantile inverts the CDF at the atoms" begin
    for p in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
        d = Geometric(p)
        for k in 0:80
            x = float(k)
            c = cdf(d, x)
            # Once rounding makes the CDF one, later outcomes cannot be recovered.
            c < 1 || break
            @test quantile(d, c) == x
            #=
              A probability strictly inside the step maps to the outcome that closes it.
              Deep in the tail the midpoint rounds onto the previous CDF value, which
              belongs to the previous outcome.
            =#
            below = cdf(d, x - 1)
            mid = (below + c) / 2
            mid > below && @test quantile(d, mid) == x
        end
    end

    # An arbitrary probability still lands on an outcome.
    for q in (0.001, 0.137, 0.5, 0.618, 0.987)
        @test isinteger(quantile(Geometric(0.3), q))
    end
end

@testset "extreme probabilities" begin
    # A vanishing success probability keeps the density and the mean finite.
    tiny = Geometric(1e-300)
    @test isfinite(logdensityof(tiny, 0.0))
    @test logdensityof(tiny, 1.0) ≈ log(1e-300)
    @test mean(tiny) ≈ 1e300
    @test cdf(tiny, 0.0) ≈ 1e-300
    @test ccdf(tiny, 0.0) ≈ 1.0

    # Almost certain success puts nearly all the mass on zero.
    nearly = Geometric(1 - 1e-16)
    @test cdf(nearly, 0.0) ≈ 1 - 1e-16
    @test quantile(nearly, 0.5) == 0.0
    @test -1e-15 < logdensityof(nearly, 0.0) < 0
    # `1 - p` rounds to half an eps, and `log1p` still resolves it.
    @test logdensityof(nearly, 1.0) ≈ log(eps(1.0) / 2) rtol = 1e-14

    # `BigFloat` agrees with `Float64` where `Float64` still resolves the value.
    for p in (0.1, 0.5, 0.9), k in (0, 1, 5, 40)
        exact = logdensityof(Geometric(BigFloat(p)), BigFloat(k))
        @test logdensityof(Geometric(p), float(k)) ≈ Float64(exact) rtol = 1e-14
    end
    # The log-CDF stays exact deep in the tail, where the CDF itself underflows.
    @test logccdf(Geometric(0.3), 3000.0) ≈ 3001 * log1p(-0.3) rtol = 1e-15
end

@testset "construction never validates" begin
    #=
      A probability above one is invalid, yet `p(1-p)^0 = p` is finite at zero. Only
      later outcomes take the logarithm of a negative number.
    =#
    invalid = Geometric(1.1)
    @test !checkparams(invalid)
    @test isfinite(logdensityof(invalid, 0.0))
    @test isnan(logdensityof(invalid, 1.0))

    for bad in (Geometric(0.0), Geometric(-0.5), Geometric(Inf), Geometric(NaN))
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, 1.0))
    end
end

@testset "sampling" begin
    d = Geometric(0.3)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Geometric(0.3f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 200_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 0.02
    @test var(draws) ≈ var(d) rtol = 0.05
end

@testset "endpoints and tails" begin
    d = Geometric(0.3)
    @test quantile(d, 1.0) == Inf
    @test isfinite(logccdf(d, 3000.0))
    @test ccdf(d, 3000.0) == 0.0

    pointmass = Geometric(1.0)
    @test logdensityof(pointmass, 0.0) == 0.0
    @test logdensityof(pointmass, 1.0) == -Inf
    @test cdf(pointmass, 0.0) == 1.0
    @test ccdf(pointmass, 0.0) == 0.0
    @test quantile(pointmass, 0.0) == 0.0
    @test quantile(pointmass, 1.0) == 0.0
    @test mean(pointmass) == 0.0
    @test var(pointmass) == 0.0
    @test entropy(pointmass) == 0.0
    @test all(iszero, rand(Xoshiro(0x524549), pointmass, 8))
end

@testset "total off the domains" begin
    d = Geometric(0.3)
    @test cdf(d, -1.0) == 0.0
    @test ccdf(d, -1.0) == 1.0
    @test logcdf(d, -1.0) == -Inf
    @test logccdf(d, -1.0) == 0.0
    @test cdf(d, 1.5) == cdf(d, 1.0)
    for x in (-1.0, 0.5, Inf, -Inf, NaN)
        @test logdensityof(d, x) == -Inf
    end
end
