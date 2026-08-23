"""
    ProbabilityMeasures

A library of normalized probability measures for probabilistic programming. It works
across numeric types, avoids allocations, and supports automatic differentiation and
GPUs.

See [`AbstractProbabilityMeasure`](@ref) for the interface a measure must satisfy,
and `ProbabilityMeasuresTest.test_measure` (in `libs/`) for the conformance suite
that checks it.
"""
module ProbabilityMeasures

using DensityInterface: DensityInterface, densityof, logdensityof
using IrrationalConstants: invsqrt2, log2π, logtwo, sqrt2
using LinearAlgebra: Diagonal, LowerTriangular, UniformScaling
using Random: Random, AbstractRNG
using SpecialFunctions: erfc, erfcinv, logerfc, loggamma
using Statistics: Statistics, cov, mean, median, quantile, std, var
using StatsAPI: StatsAPI, params

include("core/types.jl")
include("core/mathfuns.jl")
include("core/support.jl")
include("core/interface.jl")

include("univariate/continuous/normal.jl")
include("univariate/continuous/exponential.jl")
include("univariate/continuous/uniform.jl")
include("univariate/continuous/laplace.jl")

include("univariate/discrete/categorical.jl")
include("univariate/discrete/bernoulli.jl")
include("univariate/discrete/binomial.jl")
include("univariate/discrete/poisson.jl")

include("multivariate/continuous/mvnormal.jl")

# Export the operations commonly needed by probabilistic programs.

# Core types
export AbstractProbabilityMeasure
export VariateForm, Univariate, Multivariate
export ValueSupport, Continuous, Discrete
export ContinuousUnivariateMeasure, DiscreteUnivariateMeasure
export ContinuousMultivariateMeasure

# Supports
export Support, RealLine, NonNegativeReals, RealInterval, IntegerRange
export NonNegativeIntegers, RealVectors
export support, insupport

# Interface
export checkparams, validateparams, noisetype, basefloat
export cdf, ccdf, logcdf, logccdf, entropy

# Re-export common operations from package dependencies.
export logdensityof, densityof
export params
export mean, var, std, median, quantile, cov

# Measures
export Normal
export Exponential
export Uniform
export Laplace
export Categorical
export Bernoulli
export Binomial
export Poisson
export MvNormal

end
