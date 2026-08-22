"""
    VariateForm

Describes the shape of one sample: [`Univariate`](@ref) or [`Multivariate`](@ref).
"""
abstract type VariateForm end

"Draws are scalars."
struct Univariate <: VariateForm end

"Draws are vectors."
struct Multivariate <: VariateForm end

"""
    ValueSupport

Describes whether samples are [`Continuous`](@ref) or [`Discrete`](@ref).
"""
abstract type ValueSupport end

"Draws take values in a continuum, so densities are with respect to Lebesgue measure."
struct Continuous <: ValueSupport end

"Draws take values in a countable set, so densities are with respect to counting measure."
struct Discrete <: ValueSupport end

"""
    AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport}

Supertype for all probability measures.

Every subtype is normalized: its density integrates or sums to one.

# Type parameters

  - `F<:VariateForm`: the shape of a single draw.
  - `S<:ValueSupport`: whether draws are continuous or discrete.

# Required methods

  - `DensityInterface.logdensityof(d, x)`: the normalized log-density.
  - `Base.rand(rng::AbstractRNG, d)`: a single draw.
  - `Base.eltype(::Type{typeof(d)})`: the type of a draw.
  - [`support`](@ref)`(d)`.

[`insupport`](@ref), [`params`](@ref) and the moment functions all have fallbacks.

# Rules for implementations

The conformance suite (`ProbabilityMeasuresTest.test_measure` in `libs/`) checks these
rules:

 1. **Numeric types.** The result type is
    `float(promote_type(<parameter types>..., typeof(x)))`.
 2. **No errors from `logdensityof`.** Outside the support or with invalid parameters,
    return a non-finite value of the right type. Use [`checkparams`](@ref), not `isnan`,
    to detect invalid parameters.
 3. **Constructors do not validate.** See [`checkparams`](@ref).
 4. **Parameters and arguments use `Number` rather than `Real`.** This allows wrapped
    numeric values used by automatic differentiation and tracing tools.
 5. **Do not use values to drive Julia branches.** Wrapped comparisons may not produce
    a `Bool`. Use `&` and `|` for predicates and `ProbabilityMeasures.select` for a
    two-way choice. Both choices must be safe to evaluate.
"""
abstract type AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport} end

# Short names used by implementations and interface fallbacks.
const UnivariateMeasure{S} = AbstractProbabilityMeasure{Univariate,S}
const MultivariateMeasure{S} = AbstractProbabilityMeasure{Multivariate,S}
const ContinuousMeasure{F} = AbstractProbabilityMeasure{F,Continuous}
const DiscreteMeasure{F} = AbstractProbabilityMeasure{F,Discrete}
const ContinuousUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Continuous}
const DiscreteUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Discrete}
const ContinuousMultivariateMeasure = AbstractProbabilityMeasure{Multivariate,Continuous}

# Reuse the same measure for every value in a broadcast.
Base.broadcastable(d::AbstractProbabilityMeasure) = Ref(d)
