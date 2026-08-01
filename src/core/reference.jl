"""
    ReferenceMeasure

The measure that a density is taken with respect to.

This package is *normalized-only*: [`logdensityof`](@ref) returns the finished
log-density with no base-measure recursion to unroll. But the reference still has to
be recorded, because it is what a change of variables acts on and what makes a
Radon--Nikodym derivative between two measures meaningful. Keeping it as a
zero-field trait rather than a measure object gets that bookkeeping for free.
"""
abstract type ReferenceMeasure end

"Lebesgue measure on ``\\mathbb{R}^n``; the reference for continuous measures."
struct Lebesgue <: ReferenceMeasure end

"Counting measure; the reference for discrete measures."
struct Counting <: ReferenceMeasure end

"""
    reference(d) -> ReferenceMeasure

The measure that `logdensityof(d, x)` is a density with respect to.

Defaults are derived from the [`ValueSupport`](@ref) type parameter, so an ordinary
measure never writes this method. Override it only for a measure whose reference is
not implied by its support -- a mixed discrete/continuous measure, for instance.
"""
reference(::ContinuousMeasure) = Lebesgue()
reference(::DiscreteMeasure) = Counting()
