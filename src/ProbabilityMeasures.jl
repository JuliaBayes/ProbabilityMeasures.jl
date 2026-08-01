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

# Core types
export AbstractProbabilityMeasure
export VariateForm, Univariate, Multivariate, Matrixvariate
export ValueSupport, Continuous, Discrete
export UnivariateMeasure, MultivariateMeasure, MatrixvariateMeasure
export ContinuousMeasure, DiscreteMeasure
export ContinuousUnivariateMeasure, DiscreteUnivariateMeasure

# Reference measures and supports
export ReferenceMeasure, Lebesgue, Counting, reference
export Support, RealLine, PositiveReals, UnitInterval, RealInterval
export support, insupport

# Interface
export variateform, valuesupport, checkparams, noisetype, basefloat
export cdf, ccdf, logcdf, logccdf, mgf, cf
export mode, entropy, skewness, kurtosis
export zval, xval

# Re-exported so that `using ProbabilityMeasures` is enough to work with a measure.
export logdensityof, densityof
export params
export mean, var, std, median, quantile

# Measures
export Normal

end
