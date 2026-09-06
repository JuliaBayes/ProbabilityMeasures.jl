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

"The non-negative integers, ``\\{0, 1, 2, \\ldots\\}``."
struct NonNegativeIntegers <: Support end

insupport(::NonNegativeIntegers, x::Number) = isinteger(x) & (x >= zero(x))

Base.minimum(::NonNegativeIntegers) = 0
Base.maximum(::NonNegativeIntegers) = Inf

"""
    IntegerSimplex(n, k)

The non-negative integer vectors of length `k` whose entries sum to `n`.
"""
struct IntegerSimplex{N<:Integer,K<:Integer} <: Support
    n::N
    k::K
end

function insupport(s::IntegerSimplex, x::AbstractVector{<:Number})
    length(x) == s.k || return false
    remaining = s.n
    ok = true
    for xᵢ in x
        valid = isfinite(xᵢ) & isinteger(xᵢ) & (xᵢ >= zero(xᵢ)) & (xᵢ <= remaining)
        ok &= valid
        remaining -= select(valid, () -> xᵢ, () -> zero(xᵢ))
    end
    return ok & iszero(remaining)
end

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

"The unit interval, ``[0, 1]``."
struct UnitInterval <: Support end

#=
  The upper bound is tested as `1 - x >= 0` rather than `x <= 1`. A density on the
  interval takes `log(1 - x)`, whose guard compares `1 - x` with zero, and Reactant's
  pass pipeline misplaces the negation it builds when it meets that guard's complement
  written as `x <= 1`. Testing the same quantity in both places avoids the rewrite.
=#
function insupport(::UnitInterval, x::Number)
    return isfinite(x) & (x >= zero(x)) & (one(x) - x >= zero(x))
end

Base.minimum(::UnitInterval) = 0.0
Base.maximum(::UnitInterval) = 1.0

"""
    RealSimplex(k)

The non-negative real vectors of length `k` whose entries sum to one.

The sum is checked to `sqrt(eps)` of the working precision, as [`Categorical`](@ref)
checks its probabilities.
"""
struct RealSimplex <: Support
    k::Int
end

function insupport(s::RealSimplex, x::AbstractVector{<:Number})
    length(x) == s.k || return false
    total = zero(eltype(x))
    # `min` keeps a `NaN` visible without changing numeric types.
    least = zero(eltype(x))
    finite = true
    for xᵢ in x
        finite &= isfinite(xᵢ)
        total += xᵢ
        least = min(least, xᵢ)
    end
    tol = sqrt(eps(basefloat(float(eltype(x)))))
    return finite & (least >= zero(least)) & (abs(total - one(total)) <= tol)
end
