#=
  The Reactant half of the conformance suite, in an environment of its own.

  It is separate from `test/` because Reactant brings Enzyme and the XLA runtime with
  it, and the rest of the suite has no use for either. Keeping it out here also means
  Reactant's release cadence cannot gate resolution of the main environment. The
  filename avoids the `test-*.jl` pattern that `test/runtests.jl` walks for, so this
  never runs by accident from the main suite.

  Run it with `julia --project=test/reactant test/reactant/runtests.jl`, or let
  `.github/workflows/TestReactant.yml` do it.
=#
using Enzyme
using ProbabilityMeasures
using ProbabilityMeasuresTest
using Random
using Reactant
using Test

@testset "Reactant" begin
    #=
      Check the extension is loaded. Without it `test_measure` skips its Reactant
      block and the job passes having tested nothing.
    =#
    @test Base.get_extension(ProbabilityMeasures, :ProbabilityMeasuresReactantExt) !==
        nothing

    for d in (Normal(0.0, 1.0), Normal(-2.5, 0.5), Uniform(-1.0, 2.0))
        test_reactant(d, default_testpoints(d))
    end

    #=
      Enzyme through Reactant: a compiled log-likelihood differentiated with respect
      to the parameters. Reactant runs Enzyme on the MLIR and never consults
      `EnzymeRules`, so this is a different path from the `AutoEnzyme` block in the
      CPU suite, and it is why there is no `ProbabilityMeasuresReactantEnzymeExt`.

      The check is against the analytic gradient, not another AD backend.
    =#
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
