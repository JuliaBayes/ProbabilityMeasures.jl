using ProbabilityMeasures: betainc, betaincinv, logbetainc
using SpecialFunctions: beta_inc, beta_inc_inv
using Test

const SHAPES = (0.25, 0.5, 1.0, 2.5, 5.0, 50.0, 500.0)
const POINTS = (1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-9)

@testset "agreement with SpecialFunctions" begin
    for a in SHAPES, b in (0.5, 1.0, 3.0), x in POINTS
        p, q = beta_inc(a, b, x)
        got = betainc(a, b, x)
        # Both underflow to zero deep in the tail of a large first shape.
        p == 0 ? (@test got[1] == 0) : (@test got[1] ≈ p rtol = 1e-12)
        @test got[2] ≈ q rtol = 1e-12
        @test sum(got) ≈ 1
    end
end

@testset "the inverse returns to the probability it was given" begin
    for a in SHAPES, b in (0.5, 1.0, 3.0), p in (1e-300, 1e-100, 1e-12, 0.01, 0.4, 0.9)
        x, xc = betaincinv(a, b, p, 1 - p)
        (x == 0 || xc == 0) && continue
        @test x + xc ≈ 1
        got = betainc(a, b, x, xc)
        @test got[1] ≈ p rtol = 1e-11
        @test got[2] ≈ 1 - p rtol = 1e-11
    end
end

@testset "endpoints" begin
    for (a, b) in ((2.5, 0.5), (0.5, 4.0))
        @test betainc(a, b, 0.0) == (0.0, 1.0)
        @test betainc(a, b, 1.0) == (1.0, 0.0)
        @test betaincinv(a, b, 0.0, 1.0) == (0.0, 1.0)
        @test betaincinv(a, b, 1.0, 0.0) == (1.0, 0.0)
    end
end

@testset "the complement of an argument near one keeps its digits" begin
    a, b, xc = 2.5, 0.5, 1e-14
    # `1 - xc` rounds to one in `Float64`, which would lose the upper tail entirely.
    @test betainc(a, b, 1 - xc, xc)[2] > 0
    @test betainc(a, b, 1 - xc, xc)[2] ≈ beta_inc(a, b, 1 - xc, xc)[2] rtol = 1e-10
end

@testset "wider floats keep converging" begin
    setprecision(BigFloat, 256) do
        for a in (big"0.5", big"2.5", big"25.0"), b in (big"0.5", big"3.0")
            for p in (big"1e-60", big"0.01", big"0.25", big"0.9")
                x, xc = betaincinv(a, b, p, 1 - p)
                got = betainc(a, b, x, xc)
                @test abs(got[1] - p) / p < 1e-70
            end
            @test betainc(a, b, big"0.3") isa Tuple{BigFloat,BigFloat}
        end
    end

    for a in (2.5f0, 25.0f0), b in (0.5f0, 3.0f0)
        @test betainc(a, b, 0.3f0) isa Tuple{Float32,Float32}
        x, xc = betaincinv(a, b, 0.3f0, 0.7f0)
        @test x isa Float32
        @test betainc(a, b, x, xc)[1] ≈ 0.3f0 rtol = 1e-5
    end
end

@testset "the logarithm survives an underflowing tail" begin
    deep = 1e-300
    for (a, b) in ((2.5, 0.5), (0.5, 4.0), (12.0, 0.5))
        for x in (1e-12, 0.01, 0.2, 0.5)
            @test logbetainc(a, b, x, 1 - x) ≈ log(betainc(a, b, x)[1])
        end
        # `x^a` decides where the tail leaves the `Float64` range; the logarithm goes on.
        @test (betainc(a, b, deep)[1] == 0) == (a * log(deep) < log(floatmin(Float64)))
        @test logbetainc(a, b, deep, 1 - deep) ≈ a * log(deep) atol = 30
    end
    @test logbetainc(2.5, 0.5, 0.0, 1.0) == -Inf
    @test logbetainc(2.5, 0.5, 1.0, 0.0) == 0
end
