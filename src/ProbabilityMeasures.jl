"""
    ProbabilityMeasures

A library of normalized probability measures built for probabilistic programming:
type-generic, allocation-free, and clean under automatic differentiation and on the
GPU.

See [`AbstractProbabilityMeasure`](@ref) for the interface a measure must satisfy,
and `ProbabilityMeasuresTest.test_measure` (in `libs/`) for the conformance suite
that checks it.
"""
module ProbabilityMeasures

using DensityInterface: DensityInterface, densityof, logdensityof
using IrrationalConstants: invsqrt2, log2π, logtwo, sqrt2
using Random: Random, AbstractRNG
using SpecialFunctions: erfc, erfcinv, logerfc
using Statistics: Statistics, mean, median, quantile, std, var
using StatsAPI: StatsAPI, params

include("core/types.jl")
include("core/mathfuns.jl")
include("core/support.jl")
include("core/interface.jl")

include("univariate/continuous/normal.jl")
include("univariate/continuous/exponential.jl")
include("univariate/continuous/uniform.jl")

#=
  Keep exports to operations needed by PPLs. More distribution summaries, support
  types, and transform traits can be added when an implementation needs them.
=#

# Core types
export AbstractProbabilityMeasure
export VariateForm, Univariate, Multivariate
export ValueSupport, Continuous, Discrete
export ContinuousUnivariateMeasure

# Supports
export Support, RealLine, NonNegativeReals, RealInterval, support, insupport

# Interface
export checkparams, noisetype, basefloat
export cdf, ccdf, logcdf, logccdf, entropy

#=
  Re-export ecosystem functions so `using ProbabilityMeasures` is sufficient for
  common measure operations.
=#
export logdensityof, densityof
export params
export mean, var, std, median, quantile

# Measures
export Normal
export Exponential
export Uniform

end
