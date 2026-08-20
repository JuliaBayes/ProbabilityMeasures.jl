"""
    Support

Supertype for descriptions of the set a measure puts its mass on.

Supports are *singletons wherever possible*. A singleton carries no numeric payload,
so it cannot pin numeric precision and costs nothing to pass into a kernel.
"""
abstract type Support end

"The whole real line, ``(-\\infty, \\infty)``."
struct RealLine <: Support end

#=
  Add more support types when a measure needs them.
=#

"""
    support(d) -> Support

The set on which `d` places its mass.
"""
function support end

"""
    insupport(d, x) -> Bool
    insupport(s::Support, x) -> Bool

Whether `x` lies in the support.

This is a *predicate*, not a precondition. `logdensityof` is total and returns `-Inf`
outside the support on its own. This is intended for samplers, transforms, and
validation.
"""
insupport(d::AbstractProbabilityMeasure, x) = insupport(support(d), x)

insupport(::RealLine, x::Number) = isfinite(x)

Base.minimum(::RealLine) = -Inf
Base.maximum(::RealLine) = Inf

"The non-negative real line, ``[0, \\infty)``."
struct NonNegativeReals <: Support end

insupport(::NonNegativeReals, x::Number) = isfinite(x) & (x >= zero(x))

Base.minimum(::NonNegativeReals) = 0.0
Base.maximum(::NonNegativeReals) = Inf

"""
    RealInterval(a, b)

The closed interval ``[a, b]``.

"""
struct RealInterval{A<:Number,B<:Number} <: Support
    a::A
    b::B
end

insupport(s::RealInterval, x::Number) = isfinite(x) & (x >= s.a) & (x <= s.b)

Base.minimum(s::RealInterval) = s.a
Base.maximum(s::RealInterval) = s.b

"""
    RealVectors(n)

The real vectors of length `n`, ``\\mathbb{R}^n``.

"""
struct RealVectors <: Support
    n::Int
end

function insupport(s::RealVectors, x::AbstractVector{<:Number})
    return (length(x) == s.n) & all(isfinite, x)
end
