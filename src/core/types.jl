"""
    VariateForm

Trait describing the shape of a single draw from a measure: [`Univariate`](@ref) or
[`Multivariate`](@ref).

There is no `Matrixvariate`. Adding a variate form later is a non-breaking change,
so it can wait until a measure needs one.
"""
abstract type VariateForm end

"Draws are scalars."
struct Univariate <: VariateForm end

"Draws are vectors."
struct Multivariate <: VariateForm end

"""
    ValueSupport

Trait describing whether draws are [`Continuous`](@ref) or [`Discrete`](@ref).
"""
abstract type ValueSupport end

"Draws take values in a continuum, so densities are with respect to Lebesgue measure."
struct Continuous <: ValueSupport end

"Draws take values in a countable set, so densities are with respect to counting measure."
struct Discrete <: ValueSupport end

"""
    AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport}

Supertype for all probability measures.

Every subtype is a *normalized* measure: its density integrates to one against the
measure implied by its [`ValueSupport`](@ref), Lebesgue for [`Continuous`](@ref) and
counting for [`Discrete`](@ref). `logdensityof` returns the normalized value.

# Type parameters

  - `F<:VariateForm`: the shape of a single draw.
  - `S<:ValueSupport`: whether draws are continuous or discrete.

# Required methods

  - `DensityInterface.logdensityof(d, x)`: the normalized log-density.
  - `Base.rand(rng::AbstractRNG, d)`: a single draw.
  - `Base.eltype(::Type{typeof(d)})`: the type of a draw.
  - [`support`](@ref)`(d)`.

[`insupport`](@ref), [`params`](@ref) and the moment functions all have fallbacks.

# Invariants

The conformance suite (`ProbabilityMeasuresTest.test_measure`, in `libs/`) enforces
these, and they must hold:

 1. **Type genericity.** The result type is
    `float(promote_type(<parameter types>..., typeof(x)))`.
 2. **Totality.** `logdensityof` never throws. Outside the support, and for invalid
    parameters, it returns a correctly-typed non-finite value (`-Inf` or `NaN`), so it
    can be called from inside a GPU kernel. Which non-finite value comes back is not
    part of the contract; use [`checkparams`](@ref) rather than `isnan` to detect
    invalid parameters.
 3. **No validation in constructors.** See [`checkparams`](@ref).
 4. **Parameters and arguments are bounded by `Number`, not `Real`.** This admits AD
    and tracing wrappers such as Reactant's `TracedRNumber`.
 5. **No branching on a value.** A comparison between traced values is itself traced
    and cannot drive `?:`, `&&` or `||`. Use `&`/`|` for predicates and
    `ProbabilityMeasures.select` for a two-way branch, whose arms must both be total.
"""
abstract type AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport} end

#=
  Dispatch aliases used by measure implementations and interface fallbacks. Only the
  two a measure declares itself a subtype of are exported.
=#
const UnivariateMeasure{S} = AbstractProbabilityMeasure{Univariate,S}
const MultivariateMeasure{S} = AbstractProbabilityMeasure{Multivariate,S}
const ContinuousMeasure{F} = AbstractProbabilityMeasure{F,Continuous}
const DiscreteMeasure{F} = AbstractProbabilityMeasure{F,Discrete}
const ContinuousUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Continuous}
const ContinuousMultivariateMeasure = AbstractProbabilityMeasure{Multivariate,Continuous}

#=
  Treat measures as scalars during broadcast. Device broadcasts can then capture an
  `isbits` measure by value and fuse without a separate batched path.
=#
Base.broadcastable(d::AbstractProbabilityMeasure) = Ref(d)
