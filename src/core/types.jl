"""
    VariateForm

Trait describing the shape of a single draw from a measure: [`Univariate`](@ref),
[`Multivariate`](@ref), or [`Matrixvariate`](@ref).
"""
abstract type VariateForm end

"Draws are scalars."
struct Univariate <: VariateForm end

"Draws are vectors."
struct Multivariate <: VariateForm end

"Draws are matrices."
struct Matrixvariate <: VariateForm end

"""
    ValueSupport

Trait describing whether draws are [`Continuous`](@ref) or [`Discrete`](@ref). This
also selects the default reference measure; see [`reference`](@ref).
"""
abstract type ValueSupport end

"Draws take values in a continuum; the default reference is [`Lebesgue`](@ref)."
struct Continuous <: ValueSupport end

"Draws take values in a countable set; the default reference is [`Counting`](@ref)."
struct Discrete <: ValueSupport end

"""
    AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport}

Supertype for all probability measures.

Every subtype is a *normalized* measure: its density integrates to one against
[`reference`](@ref)`(d)`. There is no base-measure recursion and no unnormalized
measure in this package -- `logdensityof` returns the finished value.

# Implementing a new measure

Mandatory:

  - `DensityInterface.logdensityof(d, x)` -- the normalized log-density
  - `Base.rand(rng::AbstractRNG, d)` -- a single draw
  - `Base.eltype(::Type{typeof(d)})` -- the type of a draw
  - [`support`](@ref)`(d)`

Everything else has a fallback: [`insupport`](@ref), [`params`](@ref),
[`reference`](@ref), and the moment functions.

Three invariants are enforced by the conformance suite
(`ProbabilityMeasuresTest.test_measure`, in `libs/`) and must hold:

 1. **Type genericity.** No `Float64` literals in the density. Constants come from
    `IrrationalConstants` or `oftype`. The result type is
    `float(promote_type(<parameter types>..., typeof(x)))`.
 2. **Totality.** `logdensityof` never throws. Outside the support, and for invalid
    parameters, it returns a correctly-typed non-finite value (`-Inf` or `NaN`)
    instead. This is what makes it callable from inside a GPU kernel. Note that
    *which* non-finite value you get is not part of the contract -- use
    [`checkparams`](@ref), not `isnan`, to detect invalid parameters.
 3. **No validation in constructors.** See [`checkparams`](@ref).
"""
abstract type AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport} end

# Dispatch aliases. `AbstractProbabilityMeasure` is 28 characters, which pushes most
# `<:` clauses past the 92-column margin; prefer these in signatures.
const UnivariateMeasure{S} = AbstractProbabilityMeasure{Univariate,S}
const MultivariateMeasure{S} = AbstractProbabilityMeasure{Multivariate,S}
const MatrixvariateMeasure{S} = AbstractProbabilityMeasure{Matrixvariate,S}
const ContinuousMeasure{F} = AbstractProbabilityMeasure{F,Continuous}
const DiscreteMeasure{F} = AbstractProbabilityMeasure{F,Discrete}
const ContinuousUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Continuous}
const DiscreteUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Discrete}

"""
    variateform(d) -> Type{<:VariateForm}

The [`VariateForm`](@ref) of `d`.
"""
variateform(::Type{<:AbstractProbabilityMeasure{F}}) where {F} = F
variateform(d::AbstractProbabilityMeasure) = variateform(typeof(d))

"""
    valuesupport(d) -> Type{<:ValueSupport}

The [`ValueSupport`](@ref) of `d`.
"""
valuesupport(::Type{<:AbstractProbabilityMeasure{F,S}}) where {F,S} = S
valuesupport(d::AbstractProbabilityMeasure) = valuesupport(typeof(d))

# This single line is the whole batching story for univariate measures. Because
# measures hold scalar, `isbits` parameters, `logdensityof.(d, xs)` over a device
# array captures `d` by value and fuses into one kernel -- no wrapper type, no
# shape algebra, no separate batched code path.
Base.broadcastable(d::AbstractProbabilityMeasure) = Ref(d)
