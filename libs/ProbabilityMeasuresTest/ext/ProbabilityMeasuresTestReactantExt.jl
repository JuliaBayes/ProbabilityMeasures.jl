module ProbabilityMeasuresTestReactantExt

using ConstructionBase: constructorof
using DensityInterface: logdensityof
using ProbabilityMeasures: AbstractProbabilityMeasure, cdf, ccdf, logcdf, logccdf
using ProbabilityMeasures: checkparams
using ProbabilityMeasuresTest: ProbabilityMeasuresTest
using Random: Random
using Reactant: Reactant, @jit
using Statistics: quantile
using Test: @test, @testset

# Keep parameters in a tuple so rebuilding the measure has a known argument count.
function _traced_params(d)
    return map(Reactant.ConcreteRNumber, Tuple(ProbabilityMeasuresTest._paramvec(d)))
end

"Rebuild `d` from a tuple of wrapped parameters."
_rebuild(::D, p) where {D} = constructorof(D)(p...)

# Keep this more specific than the fallback method.
function ProbabilityMeasuresTest.test_reactant(d::AbstractProbabilityMeasure, xs)
    x = collect(float.(xs))
    rx = Reactant.to_rarray(x)

    @testset "wrapped data" begin
        @test Array(@jit(logdensityof.(d, rx))) ≈ logdensityof.(d, x)
    end

    @testset "wrapped parameters" begin
        f = (pp, xx) -> logdensityof.(_rebuild(d, pp), xx)
        @test Array(@jit(f(_traced_params(d), rx))) ≈ logdensityof.(d, x)
    end

    @testset "distribution functions" begin
        # Log-CDFs test wrapped choices; quantile tests inverse error functions.
        for f in (cdf, ccdf, logcdf, logccdf)
            @test Array(@jit(f.(d, rx))) ≈ f.(d, x)
        end

        ps = [0.01, 0.25, 0.5, 0.75, 0.99]
        rps = Reactant.to_rarray(ps)
        @test Array(@jit(quantile.(d, rps))) ≈ quantile.(d, ps)
    end

    @testset "invalid parameters" begin
        for bad in ProbabilityMeasuresTest._invalids(d)
            f = (pp, xx) -> logdensityof.(_rebuild(bad, pp), xx)
            @test !any(isfinite, Array(@jit(f(_traced_params(bad), rx))))
        end
    end

    @testset "rand" begin
        # Scalar sampling must draw plain noise even when parameters are wrapped.
        @test isfinite(Float64(@jit((() -> rand(Random.default_rng(), d))())))

        f = pp -> rand(Random.default_rng(), _rebuild(d, pp))
        @test isfinite(Float64(@jit(f(_traced_params(d)))))
    end

    @testset "checkparams" begin
        # Test the wrapped predicate through arithmetic instead of converting it to
        # a Julia boolean.
        f = (pp, xx) -> checkparams(_rebuild(d, pp)) .* xx
        @test Array(@jit(f(_traced_params(d), rx))) ≈ (checkparams(d) ? x : zero(x))
    end
end

end
