#=
  Per-measure declarations: which optional interface components a measure supports,
  which objects to test it with, and the measure-specific hooks the generic suite
  asks for.

  These live here rather than in ProbabilityMeasures itself only so that Interfaces
  stays out of the main dependency graph. If the compile-time `implements` trait
  becomes useful to downstream packages, move the `@implements` lines into the main
  package; Interfaces.jl is small enough that this would be a reasonable trade.
=#

const UNIVARIATE_OPTIONALS = (:cdf, :quantile, :mean, :var, :std, :median, :entropy)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Normal [
    Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. Invalid scales (negative and
  zero) and a non-finite location are both covered, since they fail differently.
=#
_invalids(::Normal) = (Normal(0.0, -1.0), Normal(0.0, 0.0), Normal(Inf, 1.0))
_exactparams(::Normal) = Normal(0, 1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Exponential [
    Exponential(1.0), Exponential(0.4), Exponential(3.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. A negative and a zero scale
  both fail `checkparams`, and a non-finite scale does too.
=#
_invalids(::Exponential) = (Exponential(-1.0), Exponential(0.0), Exponential(Inf))
_exactparams(::Exponential) = Exponential(1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Uniform [
    Uniform(0.0, 1.0), Uniform(-1.0, 2.0), Uniform(0.0f0, 2.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. Reversed, empty and unbounded
  intervals all fail `checkparams` and fail differently.

  The exact instance has to contain the test points of every `Uniform` the suite is
  run against, and a width other than one so that the precision check has a `log` that
  is not identically zero.
=#
_invalids(::Uniform) = (Uniform(1.0, 0.0), Uniform(0.0, 0.0), Uniform(-Inf, 1.0))
_exactparams(::Uniform) = Uniform(-1, 2)

#=
  The log-density jumps at the endpoints, so a finite-difference step wider than the
  distance from the test point to the nearest endpoint sends `test_ad`'s reference to
  `-Inf`. The default quantiles come within 0.001 of the width, which leaves that to
  the step-size heuristic; these keep a margin instead. Nothing is lost by pulling them
  in, since every distribution function here is linear, and `test-uniform.jl` checks
  the endpoints directly.
=#
function default_testpoints(d::Uniform)
    return [float(quantile(d, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
end
