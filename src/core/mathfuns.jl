"""
    select(cond, iftrue, iffalse)

Two-way branch: `iftrue()` when `cond` holds, `iffalse()` otherwise.

A `Bool` condition takes the branch, so the CPU and GPU paths cost what the
equivalent `?:` would. Tracing frontends can extend this for condition types that
cannot drive a Julia branch.

Both arms must be total because tracing may evaluate both. The thunks keep the `Bool`
path lazy.
"""
@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

"""
    logt(x)

Total `log`: returns `NaN` where `log` would throw a `DomainError`.

`log(0)` is already `-Inf`, so only negative inputs need handling.
"""
@inline function logt(x::Number)
    #=
      `log` is the arm that must stay total under tracing, where both arms evaluate.
      Real `log` throws below zero, but the traced lowering returns NaN there, and
      the select discards it either way.
    =#
    return select(x < zero(x), () -> oftype(float(x), NaN), () -> log(x))
end

"""
    erfcinvt(y)

Total `erfcinv`: returns `NaN` where `erfcinv` would throw a `DomainError`.

`quantile` must stay total (invariant 2 of [`AbstractProbabilityMeasure`](@ref)): a
probability that drifts slightly outside `[0, 1]`, for example from float noise in a `cdf`
round-trip, must not throw.
"""
@inline function erfcinvt(y::Number)
    #=
      `erfcinv`, like `log` in `logt`, is the arm that must stay total under tracing,
      where both arms evaluate. On a concrete `Bool` it is only reached when `y` is
      already in range, so the native domain check never fires.
    =#
    valid = (y >= zero(y)) & (y <= 2 * one(y))
    return select(valid, () -> erfcinv(y), () -> oftype(float(y), NaN))
end
"""
    log1pt(x)

Total `log1p`: returns `NaN` where `log1p` would throw a `DomainError`.
"""
@inline function log1pt(x::Number)
    return select(x < -one(x), () -> oftype(float(x), NaN), () -> log1p(x))
end

"""
    log1mexpt(a)

Total `log(1 - exp(a))` for `a <= 0`, accurate in both regimes: `log(-expm1(a))` near
zero, `log1p(-exp(a))` in the tail, split at `a = -log(2)`.

`a > 0` only arises from invalid parameters; `logt` keeps that branch total.
"""
@inline function log1mexpt(a::Number)
    return select(a > -logtwo, () -> logt(-expm1(a)), () -> log1p(-exp(a)))
end

"""
    basefloat(T) -> Type{<:AbstractFloat}

The plain floating-point type underlying `T`, with any AD tracking removed.

Used by [`noisetype`](@ref) to draw randomness without AD tracking. Differentiability
still enters through the measure parameters.

AD and tracing packages extend this through package extensions. The generic fallback
stops at `Real`; other `Number` wrappers require an explicit method.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{T}) where {T<:Real} = float(T)
basefloat(::Type{Bool}) = Float64
basefloat(::Type{<:Irrational}) = Float64
