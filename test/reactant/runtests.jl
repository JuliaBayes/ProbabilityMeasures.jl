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

    for d in (
        Normal(0.0, 1.0),
        Normal(-2.5, 0.5),
        Uniform(-1.0, 2.0),
        Cauchy(-1.0, 2.0),
        Geometric(0.3),
        # Both shape regimes: the log-density branches change at `α = 1`.
        Weibull(1.5, 2.0),
        Weibull(0.75, 2.5),
        # Fixed-length incomplete gamma loops and Newton steps; both shape regimes.
        Gamma(2.0, 1.5),
        Gamma(0.5, 3.0),
        # The incomplete beta fraction and the logit Newton solve; both shape regimes.
        Beta(2.0, 3.0),
        Beta(0.5, 0.5),
    )
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

        loss = (m, s) -> logdensityof(Cauchy(m, s), 2.0)
        grad = (m, s) -> Enzyme.gradient(Enzyme.Reverse, loss, m, s)
        got = @jit grad(Reactant.ConcreteRNumber(0.0), Reactant.ConcreteRNumber(1.0))
        @test [Float64(got[1]), Float64(got[2])] ≈ [0.8, 0.6] rtol = 1e-10

        # Evaluate away from zero so the `k * log(1 - p)` term contributes. The
        # gradient is `1/p - k/(1 - p)`.
        loss = p -> logdensityof(Geometric(p), 3.0)
        grad = p -> Enzyme.gradient(Enzyme.Reverse, loss, p)
        got = @jit grad(Reactant.ConcreteRNumber(0.5))
        @test Float64(got[1]) ≈ -4.0 rtol = 1e-10

        loss = (a, s) -> logdensityof(Weibull(a, s), 1.0)
        grad = (a, s) -> Enzyme.gradient(Enzyme.Reverse, loss, a, s)
        got = @jit grad(Reactant.ConcreteRNumber(1.5), Reactant.ConcreteRNumber(2.0))
        # At `z = x/θ`: `∂α = 1/α + (1 - z^α) log z` and `∂θ = (α/θ)(z^α - 1)`.
        z = 0.5
        expected = [1 / 1.5 + (1 - z^1.5) * log(z), 0.75 * (z^1.5 - 1)]
        @test [Float64(got[1]), Float64(got[2])] ≈ expected rtol = 1e-10

        #=
          The Gamma draw runs the fixed-length quantile solve, so this differentiates
          through traced loops. The shape derivative has no closed form; compare with a
          central difference of the plain-float quantile. The scale derivative is the
          unit-scale quantile itself.
        =#
        loss = (a, s) -> quantile(Gamma(a, s), 0.3)
        grad = (a, s) -> Enzyme.gradient(Enzyme.Reverse, loss, a, s)
        got = @jit grad(Reactant.ConcreteRNumber(2.0), Reactant.ConcreteRNumber(1.5))
        h = 1e-6
        dα = (quantile(Gamma(2.0 + h, 1.5), 0.3) - quantile(Gamma(2.0 - h, 1.5), 0.3)) / 2h
        @test Float64(got[1]) ≈ dα rtol = 1e-6
        @test Float64(got[2]) ≈ quantile(Gamma(2.0, 1.0), 0.3) rtol = 1e-10
    end
end
