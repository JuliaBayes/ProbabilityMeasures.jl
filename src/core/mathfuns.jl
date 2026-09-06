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
    logistic(x)

The logistic function ``1/(1 + e^{-x})``.

It saturates instead of throwing or overflowing: once `exp(-x)` overflows, past `|x|`
of about `709` in `Float64` and `88` in `Float32`, the result is exactly one or exactly
zero.
"""
@inline logistic(x::Number) = inv(one(x) + exp(-x))

"""
    logbetat(a, b)

Compute ``\\log B(a, b)``, returning `NaN` instead of throwing for a non-positive
argument.

`loggamma` throws for a negative non-integer, so a measure with an invalid shape would
otherwise raise a `DomainError` rather than return a non-finite density.

The arguments are typed separately: the numbers a reverse-mode tape produces need not
share a type with the ones it was given.
"""
@inline function logbetat(a::Number, b::Number)
    positive = (a > zero(a)) & (b > zero(b))
    # Some tools evaluate both choices, so call `loggamma` only with valid arguments.
    return select(
        positive,
        () -> loggamma(a) + loggamma(b) - loggamma(a + b),
        () -> oftype(float(a + b), NaN),
    )
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
