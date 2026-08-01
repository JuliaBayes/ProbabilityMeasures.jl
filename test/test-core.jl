using ProbabilityMeasures
using ProbabilityMeasures: logt, sqrtt, _promoted_paramtype
using ForwardDiff: ForwardDiff
using Test

@testset "traits" begin
    d = Normal(0.0, 1.0)
    @test variateform(d) === Univariate
    @test valuesupport(d) === Continuous
    @test d isa ContinuousUnivariateMeasure
    @test d isa UnivariateMeasure
    @test d isa ContinuousMeasure
    @test !(d isa DiscreteMeasure)
end

@testset "reference defaults from ValueSupport" begin
    # No measure should ever have to write a `reference` method; it falls out of the
    # ValueSupport type parameter.
    @test reference(Normal(0.0, 1.0)) === Lebesgue()
    @test Lebesgue() isa ReferenceMeasure
    @test Counting() isa ReferenceMeasure
end

@testset "support" begin
    @test support(Normal(0.0, 1.0)) === RealLine()
    @test insupport(Normal(0.0, 1.0), 0.0)
    @test !insupport(Normal(0.0, 1.0), Inf)
    @test !insupport(Normal(0.0, 1.0), NaN)

    @test insupport(PositiveReals(), 1.0)
    @test !insupport(PositiveReals(), 0.0)
    @test insupport(UnitInterval(), 0.0)
    @test insupport(UnitInterval(), 1.0)
    @test !insupport(UnitInterval(), 1.5)

    s = RealInterval(-1.0, 2.0)
    @test insupport(s, 0.0)
    @test !insupport(s, 3.0)
    @test minimum(s) == -1.0
    @test maximum(s) == 2.0

    # Singleton supports carry no numeric payload, so they cannot pin a precision.
    @test isbits(RealLine())
    @test sizeof(RealLine()) == 0
end

@testset "total math functions" begin
    # These exist so `logdensityof` can never throw; see the totality invariant.
    @test isnan(logt(-1.0))
    @test logt(0.0) == -Inf
    @test logt(1.0) == 0.0
    @test logt(-1.0f0) isa Float32
    @test isnan(sqrtt(-1.0))
    @test sqrtt(4.0) == 2.0

    # Base throws on both of these; that is the behaviour being replaced.
    @test_throws DomainError log(-1.0)
    @test_throws DomainError sqrt(-1.0)
end

@testset "basefloat strips AD tracking" begin
    @test basefloat(Float32) === Float32
    @test basefloat(Float64) === Float64
    @test basefloat(BigFloat) === BigFloat
    @test basefloat(Int) === Float64
    @test basefloat(Bool) === Float64

    # The ForwardDiff extension is what makes reparameterized sampling work.
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
