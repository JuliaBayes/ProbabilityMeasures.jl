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
include("core/reference.jl")
include("core/interface.jl")

include("univariate/continuous/normal.jl")

#=
  Every exported name is one a PPL is expected to call. That is a deliberate
  constraint rather than an aesthetic one: adding an export later is a non-breaking
  change, removing one is not, so anything speculative costs more to ship now than
  to withhold.

  Deliberately absent, and easy to add when something needs them: `mode`,
  `skewness`, `kurtosis`, `mgf`, `cf` (Distributions.jl inheritance, not inference);
  `Matrixvariate`; `PositiveReals`/`UnitInterval`/`RealInterval` (no measure uses
  them yet); `variateform`/`valuesupport` (the type parameters are already there).
=#

# Core types
export AbstractProbabilityMeasure
export VariateForm, Univariate, Multivariate
export ValueSupport, Continuous, Discrete
export ContinuousUnivariateMeasure

# Reference measures and supports
export ReferenceMeasure, Lebesgue, Counting, reference
export Support, RealLine, support, insupport

# Interface
export checkparams, noisetype, basefloat
export cdf, ccdf, logcdf, logccdf, entropy

# Re-exported so that `using ProbabilityMeasures` is enough to work with a measure.
# These are existing ecosystem names, not new API surface.
export logdensityof, densityof
export params
export mean, var, std, median, quantile

# Measures
export Normal

end
