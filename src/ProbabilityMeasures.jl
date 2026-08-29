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
using IrrationalConstants: invsqrt2, log2π, logπ, logtwo, sqrt2
using LinearAlgebra: Diagonal, LowerTriangular, UniformScaling
using Random: Random, AbstractRNG
using SpecialFunctions: digamma, erfc, erfcinv, gamma_inc, logbeta, logerfc, loggamma
using Statistics: Statistics, cov, mean, median, quantile, std, var
using StatsAPI: StatsAPI, params

include("core/types.jl")
include("core/mathfuns.jl")
include("core/betainc.jl")
include("core/support.jl")
include("core/interface.jl")

include("univariate/continuous/normal.jl")
include("univariate/continuous/lognormal.jl")
include("univariate/continuous/exponential.jl")
include("univariate/continuous/uniform.jl")
include("univariate/continuous/laplace.jl")
include("univariate/continuous/cauchy.jl")
include("univariate/continuous/tdist.jl")

include("univariate/discrete/categorical.jl")
include("univariate/discrete/bernoulli.jl")
include("univariate/discrete/binomial.jl")
include("univariate/discrete/poisson.jl")

include("multivariate/continuous/locationscale.jl")
include("multivariate/continuous/mvnormal.jl")
include("multivariate/continuous/mvtdist.jl")
include("multivariate/discrete/multinomial.jl")

# Export the operations commonly needed by probabilistic programs.

# Core types
export AbstractProbabilityMeasure
export VariateForm, Univariate, Multivariate
export ValueSupport, Continuous, Discrete
export ContinuousUnivariateMeasure, DiscreteUnivariateMeasure
export ContinuousMultivariateMeasure, DiscreteMultivariateMeasure

# Supports
export Support,
    RealLine,
    NonNegativeReals,
    PositiveReals,
    NonNegativeIntegers,
    RealInterval,
    IntegerRange,
    IntegerSimplex,
    RealVectors
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
export LogNormal
export Exponential
export Uniform
export Laplace
export Cauchy
export TDist
export Categorical
export Bernoulli
export Binomial
export Poisson
export MvNormal
export MvTDist
export Multinomial

end
