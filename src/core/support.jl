"""
    Support

Describes the values a measure can produce.
"""
abstract type Support end

"The whole real line, ``(-\\infty, \\infty)``."
struct RealLine <: Support end

"""
    support(d) -> Support

The set on which `d` places its mass.
"""
function support end

"""
    insupport(d, x) -> Bool
    insupport(s::Support, x) -> Bool

Whether `x` lies in the support.

`logdensityof` handles values outside the support on its own. Use `insupport` when a
sampler, transform, or validation step needs a boolean check.
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

"The positive real line, ``(0, \\infty)``."
struct PositiveReals <: Support end

insupport(::PositiveReals, x::Number) = isfinite(x) & (x > zero(x))

Base.minimum(::PositiveReals) = 0.0
Base.maximum(::PositiveReals) = Inf

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
    IntegerRange(a, b)

The consecutive integers ``\\{a, a+1, \\ldots, b\\}``.

`b < a` describes the empty set.
"""
struct IntegerRange{A<:Integer,B<:Integer} <: Support
    a::A
    b::B
end

insupport(s::IntegerRange, x::Number) = isinteger(x) & (x >= s.a) & (x <= s.b)

Base.minimum(s::IntegerRange) = s.a
Base.maximum(s::IntegerRange) = s.b

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
