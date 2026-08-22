# Reactant has its own test environment because it brings large compiler dependencies.
# Run with `julia --project=test/reactant test/reactant/runtests.jl`.
using Enzyme
using ProbabilityMeasures
using ProbabilityMeasuresTest
using Random
using Reactant
using Test

@testset "Reactant" begin
    # Fail rather than silently skipping every Reactant check.
    @test Base.get_extension(ProbabilityMeasures, :ProbabilityMeasuresReactantExt) !==
        nothing

    for d in (Normal(0.0, 1.0), Normal(-2.5, 0.5), Uniform(-1.0, 2.0))
        test_reactant(d, default_testpoints(d))
    end

    # Differentiate compiled Reactant code with Enzyme and compare with the formula.
    @testset "Enzyme gradient" begin
        xs = randn(Xoshiro(42), 8)
        μ, σ = 0.5, 1.5
        analytic = [sum((xs .- μ) ./ σ^2), sum(@. (xs - μ)^2 / σ^3 - 1 / σ)]

        loss = (m, s, x) -> sum(logdensityof.(Normal(m, s), x))
        grad = (m, s, x) -> Enzyme.gradient(Enzyme.Reverse, loss, m, s, Enzyme.Const(x))
        got = @jit grad(
            Reactant.ConcreteRNumber(μ), Reactant.ConcreteRNumber(σ), Reactant.to_rarray(xs)
        )

        @test [Float64(got[1]), Float64(got[2])] ≈ analytic rtol = 1e-10
    end
end
