using ProbabilityMeasures
using ProbabilityMeasures: logt, _promoted_paramtype
using ProbabilityMeasures: UnivariateMeasure, ContinuousMeasure, DiscreteMeasure
using ForwardDiff: ForwardDiff
using Random: Random
using Test

@testset "traits" begin
    d = Normal(0.0, 1.0)
    # The type parameters expose both traits through `isa`.
    @test d isa AbstractProbabilityMeasure{Univariate,Continuous}
    @test d isa ContinuousUnivariateMeasure
    @test d isa UnivariateMeasure
    @test d isa ContinuousMeasure
    @test !(d isa DiscreteMeasure)

    k = Categorical([0.5, 0.5])
    @test k isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test k isa DiscreteUnivariateMeasure
    @test k isa DiscreteMeasure
    @test !(k isa ContinuousMeasure)
end

@testset "support" begin
    @test support(Normal(0.0, 1.0)) === RealLine()
    @test insupport(Normal(0.0, 1.0), 0.0)
    @test !insupport(Normal(0.0, 1.0), Inf)
    @test !insupport(Normal(0.0, 1.0), NaN)

    @test minimum(RealLine()) == -Inf
    @test maximum(RealLine()) == Inf

    # `RealLine` stores no numeric value that could affect precision.
    @test isbits(RealLine())
    @test sizeof(RealLine()) == 0

    p = PositiveReals()
    @test minimum(p) == 0.0
    @test maximum(p) == Inf
    @test insupport(p, 1.0)
    @test !insupport(p, 0.0)
    @test !insupport(p, -1.0)
    @test !insupport(p, Inf)
    @test !insupport(p, NaN)

    s = RealInterval(-1.0f0, 2)
    @test s isa RealInterval{Float32,Int}
    @test minimum(s) === -1.0f0
    @test maximum(s) === 2
    @test insupport(s, 0.0)
    @test insupport(s, -1.0f0)
    @test !insupport(s, 2.5)
    @test !insupport(s, Inf)
    @test isbits(s)

    r = IntegerRange(1, 3)
    @test minimum(r) === 1
    @test maximum(r) === 3
    @test insupport(r, 2)
    @test insupport(r, 2.0)
    @test !insupport(r, 2.5)
    @test !insupport(r, 0)
    @test !insupport(r, 4)
    @test !insupport(r, NaN)
    @test !insupport(r, Inf)
    @test isbits(r)

    v = RealVectors(2)
    @test isbits(v)
    @test insupport(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), [0.0, 1.0])
    @test insupport(v, [0.0, 1.0])
    @test !insupport(v, [0.0])
    @test !insupport(v, [0.0, 0.0, 0.0])
    @test !insupport(v, [0.0, Inf])

    # Vector support has no scalar bounds.
    @test_throws MethodError minimum(v)
    @test_throws MethodError maximum(v)
end

@testset "total math functions" begin
    # These helpers return non-finite values instead of throwing.
    @test isnan(logt(-1.0))
    @test logt(0.0) == -Inf
    @test logt(1.0) == 0.0
    @test logt(-1.0f0) isa Float32

    @test_throws DomainError log(-1.0)
end

@testset "basefloat returns the plain type" begin
    @test basefloat(Float32) === Float32
    @test basefloat(Float64) === Float64
    @test basefloat(BigFloat) === BigFloat
    @test basefloat(Int) === Float64
    @test basefloat(Bool) === Float64

    # The ForwardDiff extension finds the plain type inside a dual number.
    @test basefloat(ForwardDiff.Dual{Nothing,Float32,1}) === Float32
    @test noisetype(Normal(ForwardDiff.Dual(0.0, 1.0), 1.0)) === Float64
    @test noisetype(Normal(0.0f0, 1.0f0)) === Float32
end

@testset "params" begin
    d = Normal(1.5, 2.5)
    @test params(d) === (μ=1.5, σ=2.5)
    @test keys(params(d)) === (:μ, :σ)
    @test _promoted_paramtype(typeof(Normal(0.0f0, 1))) === Float32
end

@testset "validateparams throws where checkparams says no" begin
    d = Normal(1.5, 2.5)
    # Valid input is returned unchanged.
    @test validateparams(d) === d
    @test validateparams(Categorical([0.5, 0.5])) isa Categorical

    @test_throws DomainError validateparams(Normal(0.0, -1.0))
    @test_throws DomainError validateparams(Exponential(-1.0))
    @test_throws DomainError validateparams(Uniform(1.0, 0.0))
    @test_throws DomainError validateparams(MvNormal([0.0], [-1.0;;]))

    # An unnormalized vector can still give finite densities.
    unnormalized = Categorical([2.0, 2.0])
    @test isfinite(logdensityof(unnormalized, 1.0))
    @test logdensityof(unnormalized, 1.0) ≈
        logdensityof(Categorical([0.5, 0.5]), 1.0) + log(4)
    @test_throws DomainError validateparams(unnormalized)
end

@testset "measures broadcast as scalars" begin
    d = Normal(0.0, 1.0)
    xs = [-1.0, 0.0, 1.0]
    @test Base.broadcastable(d) isa Base.RefValue
    @test logdensityof.(d, xs) == [logdensityof(d, x) for x in xs]
    @test length(logdensityof.([Normal(0.0, 1.0), Normal(1.0, 1.0)], 0.5)) == 2
end

#=
  `rand(rng, T)` can return exactly zero. Inverse-CDF samplers built on `log(u)` would
  return `Inf` there, so they use `log1p(-u)` instead. Cauchy is left out: both ends of
  its quantile are infinite, so a zero draw is its true quantile.
=#
struct ZeroRNG <: Random.AbstractRNG end
Random.rand(::ZeroRNG, ::Type{T}) where {T<:AbstractFloat} = zero(T)

@testset "samplers survive an exact zero draw" begin
    for d in (Exponential(2.0), Laplace(1.0, 2.0), Weibull(0.75, 2.5), Geometric(0.3))
        x = rand(ZeroRNG(), d)
        @test isfinite(x)
        @test insupport(d, x)
    end
end
