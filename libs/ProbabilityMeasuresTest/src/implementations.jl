#=
  Per-measure declarations: which optional interface components a measure supports,
  which objects to test it with, and the measure-specific hooks the generic suite
  asks for.

  These live here rather than in ProbabilityMeasures itself only so that Interfaces
  stays out of the main dependency graph. If the compile-time `implements` trait
  becomes useful to downstream packages, move the `@implements` lines into the main
  package -- Interfaces.jl is small enough that this would be a reasonable trade.
=#

const UNIVARIATE_OPTIONALS = (:cdf, :quantile, :mean, :var, :std, :median, :entropy)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Normal [
    Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0)
]

# Hooks used by `test_totality` and `test_genericity`. Both an invalid scale (NaN)
# and a non-finite location (-Inf) are covered, since they fail differently.
_invalids(::Normal) = (Normal(0.0, -1.0), Normal(0.0, 0.0), Normal(Inf, 1.0))
_exactparams(::Normal) = Normal(0, 1)
