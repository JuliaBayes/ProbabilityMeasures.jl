"""
    select(cond, iftrue, iffalse)

Call `iftrue()` when `cond` is true and `iffalse()` otherwise.

With a `Bool`, only the selected function is called. Tools with wrapped conditions can
add methods that evaluate both, so each function must be safe to call.
"""
@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

"""
    wrappedconditions(T) -> Bool

Whether comparisons on a `T` give a wrapped condition rather than a `Bool`.

A loop that stops when its terms stop changing the result needs a `Bool`. Where this is
true, such loops run a fixed number of terms instead, with [`select`](@ref) in place of
every value-driven branch. It is false for every type Base knows; the Reactant extension
sets it for traced numbers.
"""
wrappedconditions(::Type) = false
    pick(cond, iftrue, iffalse)

[`select`](@ref) between two values that are already computed.

Both values are evaluated whatever `cond` is, so use this only where evaluating both is
safe and cheap. Passing them as arguments also keeps them out of the caller's closures:
a captured variable that a loop reassigns is boxed, which costs inference and an
allocation.
"""
@inline pick(cond, iftrue, iffalse) = select(cond, () -> iftrue, () -> iffalse)

"""
    logt(x)

Like `log`, but returns `NaN` instead of throwing for a negative input.

`log(0)` is already `-Inf`, so only negative inputs need handling.
"""
@inline function logt(x::Number)
    # Some tools evaluate both choices, so neither choice may throw.
    return select(x < zero(x), () -> oftype(float(x), NaN), () -> log(x))
end

"""
    erfcinvt(y)

Like `erfcinv`, but returns `NaN` instead of throwing outside its domain.

Rounding can move a probability slightly outside `[0, 1]`, so quantiles need this
version to return `NaN` instead of throwing.
"""
@inline function erfcinvt(y::Number)
    # Some tools evaluate both choices, so call `erfcinv` only with a valid value.
    valid = (y >= zero(y)) & (y <= 2 * one(y))
    return select(valid, () -> erfcinv(y), () -> oftype(float(y), NaN))
end
"""
    log1pt(x)

Like `log1p`, but returns `NaN` instead of throwing below -1.
"""
@inline function log1pt(x::Number)
    return select(x < -one(x), () -> oftype(float(x), NaN), () -> log1p(x))
end

"""
    log1mexpt(a)

Compute `log(1 - exp(a))` accurately for `a <= 0`. Use `log(-expm1(a))` near zero
and `log1p(-exp(a))` in the tail.

When invalid parameters make `a > 0`, `logt` returns `NaN` instead of throwing.
"""
@inline function log1mexpt(a::Number)
    return select(a > -logtwo, () -> logt(-expm1(a)), () -> log1p(-exp(a)))
end

"""
    basefloat(T) -> Type{<:AbstractFloat}

The plain floating-point type inside `T`.

[`noisetype`](@ref) uses it for random noise. Package extensions add methods for
wrapped numeric types.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{T}) where {T<:Real} = float(T)
basefloat(::Type{Bool}) = Float64
basefloat(::Type{<:Irrational}) = Float64

"""
    xlogyt(x, y)

Compute `x * log(y)`, taking the product to be zero when `x` is zero and `log(y)` is
not finite.

A density whose exponent is zero needs that convention at the edge of its support,
where the plain product would be `NaN`. The guard tests `log(y)` rather than `x` alone
so that a zero `x` away from the edge still returns `x * log(y)`: that product is zero
either way, but only the product carries the derivative of `x`, which a shape of
exactly one would otherwise lose.
"""
@inline function xlogyt(x::Number, y::Number)
    ly = logt(y)
    # Some tools evaluate both choices, so neither choice may throw.
    return select((x == zero(x)) & !isfinite(ly), () -> zero(float(x * ly)), () -> x * ly)
end

"""
    logbetat(α, β)

Compute ``\\log B(\\alpha, \\beta)`` from log-gamma values, returning `NaN` instead of
throwing for a non-positive argument.

`SpecialFunctions.logbeta` branches on its arguments and so cannot be traced. The three
log-gamma calls here can.
"""
@inline function logbetat(α::Number, β::Number)
    valid = (α > zero(α)) & (β > zero(β))
    # `loggamma` throws below zero, so call it at one when the arguments are invalid.
    a = pick(valid, α, one(α))
    b = pick(valid, β, one(β))
    v = loggamma(a) + loggamma(b) - loggamma(a + b)
    return pick(valid, v, oftype(v, NaN))
end
