using ProbabilityMeasures
using ProbabilityMeasures: logt, _promoted_paramtype
using ProbabilityMeasures: UnivariateMeasure, ContinuousMeasure, DiscreteMeasure
using ForwardDiff: ForwardDiff
using Test

@testset "traits" begin
    d = Normal(0.0, 1.0)
    #=
      There is no `variateform`/`valuesupport` accessor pair: the type parameters are
      already on the type, and `isa` is the idiom.
    =#
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

    # Singleton supports carry no numeric payload, so they cannot pin a precision.
    @test isbits(RealLine())
    @test sizeof(RealLine()) == 0

    # Interval endpoints keep their original types.
    s = RealInterval(-1.0f0, 2)
    @test s isa RealInterval{Float32,Int}
    @test minimum(s) === -1.0f0
    @test maximum(s) === 2
    @test insupport(s, 0.0)
    @test insupport(s, -1.0f0)
    @test !insupport(s, 2.5)
    @test !insupport(s, Inf)
    @test isbits(s)

    #=
      `IntegerRange` keeps `Integer` endpoints even where the draws are floats, so the
      support can be iterated. A non-integer argument is outside it.
    =#
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

    # Vector support checks both length and values.
    v = RealVectors(2)
    @test isbits(v)
    @test insupport(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), [0.0, 1.0])
    @test insupport(v, [0.0, 1.0])
    @test !insupport(v, [0.0])
    @test !insupport(v, [0.0, 0.0, 0.0])
    @test !insupport(v, [0.0, Inf])

    # Vector support has no scalar integration bounds.
    @test_throws MethodError minimum(v)
    @test_throws MethodError maximum(v)
end

@testset "total math functions" begin
    # These exist so `logdensityof` can never throw; see the totality invariant.
    @test isnan(logt(-1.0))
    @test logt(0.0) == -Inf
    @test logt(1.0) == 0.0
    @test logt(-1.0f0) isa Float32

    # Base throws on the same input.
    @test_throws DomainError log(-1.0)
end

@testset "basefloat strips AD tracking" begin
    @test basefloat(Float32) === Float32
    @test basefloat(Float64) === Float64
    @test basefloat(BigFloat) === BigFloat
    @test basefloat(Int) === Float64
    @test basefloat(Bool) === Float64

    # Reparameterized sampling relies on the ForwardDiff extension.
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

@testset "measures broadcast as scalars" begin
    d = Normal(0.0, 1.0)
    xs = [-1.0, 0.0, 1.0]
    @test Base.broadcastable(d) isa Base.RefValue
    @test logdensityof.(d, xs) == [logdensityof(d, x) for x in xs]
    # Broadcasting two measures against one point works too.
    @test length(logdensityof.([Normal(0.0, 1.0), Normal(1.0, 1.0)], 0.5)) == 2
end
