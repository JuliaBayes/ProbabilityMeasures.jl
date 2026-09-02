"""
    select(cond, iftrue, iffalse)

Call `iftrue()` when `cond` is true and `iffalse()` otherwise.

With a `Bool`, only the selected function is called. Tools with wrapped conditions can
add methods that evaluate both, so each function must be safe to call.
"""
@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

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
    basevalue(x) -> AbstractFloat

The plain floating-point value inside `x`.

A rejection sampler compares against it so that its accept step runs on plain numbers
and stays independent of the wrapped numeric types automatic differentiation and
tracing systems substitute. The accepted noise then enters the draw through arithmetic
on the parameters, which is what makes the draw differentiable. Package extensions add
methods for wrapped types.
"""
basevalue(x::Number) = float(x)
