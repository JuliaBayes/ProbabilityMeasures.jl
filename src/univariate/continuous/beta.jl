"""
    Beta(α, β)
    Beta()

The beta measure on ``[0, 1]`` with shapes `α` and `β`. Its density is

```math
p(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha, \\beta)}
```

`Beta()` creates the uniform measure on ``[0, 1]`` using `Float64` values.

# Arguments

  - `α::Number`: the shape pulling mass towards one.
  - `β::Number`: the shape pulling mass towards zero.

Both shapes must be finite and positive. The constructor does not check them; use
[`validateparams`](@ref) for user input.

Density results follow Julia's promotion rules for `α`, `β`, and the evaluation point.
An `α` below one makes the density infinite at zero, and a `β` below one makes it
infinite at one. That is the value there, not an error.

`cdf`, `ccdf`, `quantile`, and `rand` use `SpecialFunctions.beta_inc` for the
floating-point types it has methods for, listed in [`BetaTailFloat`](@ref). Every other
type, `BigFloat` and the wrapped numbers differentiation and tracing tools substitute
included, goes through [`incbeta`](@ref), a continued fraction of fixed length that is
inverted by bracketed Newton steps. A draw is the quantile of a uniform draw on either
path, so it carries a derivative with respect to both shapes.

The fraction matches the closed form to about `1e-14` for shapes between `0.01` and
`1000`. Inverting it is accurate to `1e-12` for shapes between `0.1` and `100` and
loses accuracy outside that range, reaching `3e-5` near shape `0.01`, so keep the
shapes moderate when differentiating or tracing a quantile.

Differentiate with ForwardDiff or Enzyme. Zygote and Mooncake rewrite Julia source
instead of wrapping numeric types, so they reach `beta_inc` rather than the fraction,
and `SpecialFunctions` carries no differentiation rule for it: they return a zero
derivative with respect to `β` rather than an error. ForwardDiff reaches the fraction,
and Enzyme differentiates the compiled code, so both are correct.

Unlike the other univariate measures, `rand` allocates: `beta_inc` allocates on its
asymptotic branches.
"""
struct Beta{A<:Number,B<:Number} <: ContinuousUnivariateMeasure
    α::A
    β::B
end

Beta() = Beta(1.0, 1.0)

Base.eltype(::Type{Beta{A,B}}) where {A,B} = float(promote_type(A, B))

function checkparams(d::Beta)
    return (d.α > zero(d.α)) & (d.β > zero(d.β)) & isfinite(d.α) & isfinite(d.β)
end

support(::Beta) = UnitInterval()

"""
    betatype(d::Beta, x)

The floating-point type of a density or probability computed from `d` and `x`.
"""
@inline function betatype(d::Beta, x::Number)
    return float(promote_type(typeof(d.α), typeof(d.β), typeof(x)))
end

@inline function DensityInterface.logdensityof(d::Beta, x::Number)
    T = betatype(d, x)
    α, β, y = convert(T, d.α), convert(T, d.β), convert(T, x)
    value = xlogyt(α - one(T), y) + xlogyt(β - one(T), one(T) - y) - logbetat(α, β)
    return pick(insupport(d, x) & checkparams(d), value, convert(T, -Inf))
end

# Draw plain noise and apply the quantile, so the shapes stay in the result.
@inline function Base.rand(rng::AbstractRNG, d::Beta)
    return quantile(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Beta) = d.α / (d.α + d.β)

function Statistics.var(d::Beta)
    s = d.α + d.β
    return d.α * d.β / (s^2 * (s + one(s)))
end

function entropy(d::Beta)
    T = eltype(d)
    α, β = convert(T, d.α), convert(T, d.β)
    s = α + β
    valid = checkparams(d)
    # `digamma` has poles at the non-positive integers, so evaluate it at one instead.
    a = pick(valid, α, one(T))
    b = pick(valid, β, one(T))
    h =
        logbetat(α, β) - (α - one(T)) * digamma(a) - (β - one(T)) * digamma(b) +
        (s - 2 * one(T)) * digamma(a + b)
    return pick(valid, h, convert(T, NaN))
end

"""
    BetaTailFloat

The floating-point types `SpecialFunctions.beta_inc` has methods for.

Every other type takes the continued fraction in [`incbeta`](@ref).
"""
const BetaTailFloat = Union{Float16,Float32,Float64}

#=
  `betatype` picks the path, as it does for `Poisson`. The closed form is the more
  accurate of the two for extreme shapes, and the only one whose error is bounded by
  someone else's work, so use it wherever it has a method.
=#
cdf(d::Beta, x::Number) = _cdf(betatype(d, x), d, x)
ccdf(d::Beta, x::Number) = _ccdf(betatype(d, x), d, x)
Statistics.quantile(d::Beta, q::Number) = _quantile(betatype(d, q), d, q)

# `beta_inc` returns both tails, so neither is left to a subtraction from one.
@inline function _tails(::Type{T}, d::Beta, x::Number) where {T<:BetaTailFloat}
    nan = convert(T, NaN)
    checkparams(d) || return (nan, nan)
    y = convert(T, x)
    isnan(y) && return (nan, nan)
    y <= zero(T) && return (zero(T), one(T))
    y >= one(T) && return (one(T), zero(T))
    return beta_inc(convert(T, d.α), convert(T, d.β), y)
end

_cdf(::Type{T}, d::Beta, x::Number) where {T<:BetaTailFloat} = _tails(T, d, x)[1]
_ccdf(::Type{T}, d::Beta, x::Number) where {T<:BetaTailFloat} = _tails(T, d, x)[2]

function _quantile(::Type{T}, d::Beta, q::Number) where {T<:BetaTailFloat}
    checkparams(d) || return convert(T, NaN)
    p = convert(T, q)
    # A `NaN` fails both comparisons and leaves here too.
    ((p >= zero(T)) & (p <= one(T))) || return convert(T, NaN)
    return beta_inc_inv(convert(T, d.α), convert(T, d.β), p)[1]
end

"""
    cfterms(T)

The number of terms [`incbeta`](@ref) evaluates at the precision of `T`.

A convergence test would branch on a value, so the count is fixed instead. Against
`SpecialFunctions.beta_inc` over shapes from `0.01` to `1000`, the fraction settles by
about 50 terms at `Float64` precision, so one term per bit plus 16 leaves margin.
Running past convergence costs time and drifts the result by about `1e-15`.
"""
@inline cfterms(::Type{T}) where {T<:Number} = precision(basefloat(T)) + 16

"""
    bisections(T)

The number of times [`_invert`](@ref) halves its bracket at the precision of `T`.

Halving past the working resolution collapses the bracket onto a single value. The
Newton steps that follow reject any candidate outside their bracket, so a collapsed one
rejects every candidate and leaves the root at a midpoint, which is constant in the
shapes and carries no derivative. Stopping eight bits short leaves Newton room to work.
"""
@inline bisections(::Type{T}) where {T<:Number} = min(40, precision(basefloat(T)) - 8)

# Lentz's algorithm divides by its running denominators, so hold them away from zero.
@inline function nonzero(v::T) where {T<:Number}
    tiny = convert(T, floatmin(basefloat(T)))
    return pick(abs(v) < tiny, tiny, v)
end

"""
    incbeta(α, β, x)

The regularized incomplete beta function ``I_x(\\alpha, \\beta)``, which is the
probability a `Beta(α, β)` draw falls at or below `x`.

The continued fraction behind it converges quickly only below the measure's centre, so
points above it use the reflection ``I_x(\\alpha, \\beta) = 1 - I_{1-x}(\\beta,
\\alpha)``. The term count comes from the precision of the arguments rather than from a
convergence test, so control flow never depends on a value.
"""
@inline incbeta(α::Number, β::Number, x::Number) = _incbeta(promote(α, β, x)...)

@inline function _incbeta(α::T, β::T, x::T) where {T<:Number}
    # Reflect above the point where the fraction stops converging quickly.
    reflect = x * (α + β + 2 * one(T)) > α + one(T)
    a = pick(reflect, β, α)
    b = pick(reflect, α, β)
    y = pick(reflect, one(T) - x, x)
    front = exp(xlogyt(a, y) + xlogyt(b, one(T) - y) - logbetat(a, b)) / a
    v = front * betacf(a, b, y)
    return pick(reflect, one(T) - v, v)
end

"""
    betacf(a, b, x)

The continued fraction of [`incbeta`](@ref), evaluated by Lentz's algorithm.
"""
@inline function betacf(a::T, b::T, x::T) where {T<:Number}
    qab, qap, qam = a + b, a + one(T), a - one(T)
    c = one(T)
    d = one(T) / nonzero(one(T) - qab * x / qap)
    h = d
    for m in 1:cfterms(T)
        n = convert(T, m)
        m2 = 2 * n
        # The even term.
        aa = n * (b - n) * x / ((qam + m2) * (a + m2))
        d = one(T) / nonzero(one(T) + aa * d)
        c = nonzero(one(T) + aa / c)
        h *= d * c
        # The odd term.
        aa = -(a + n) * (qab + n) * x / ((a + m2) * (qap + m2))
        d = one(T) / nonzero(one(T) + aa * d)
        c = nonzero(one(T) + aa / c)
        h *= d * c
    end
    return h
end

# Both tails come from the same fraction, each in the orientation that computes it
# directly rather than by subtracting the other from one.
function _cdf(::Type{T}, d::Beta, x::Number) where {T}
    α, β = convert(T, d.α), convert(T, d.β)
    y = clamp(convert(T, x), zero(T), one(T))
    return pick(checkparams(d), incbeta(α, β, y), convert(T, NaN))
end

function _ccdf(::Type{T}, d::Beta, x::Number) where {T}
    α, β = convert(T, d.α), convert(T, d.β)
    y = clamp(convert(T, x), zero(T), one(T))
    return pick(checkparams(d), incbeta(β, α, one(T) - y), convert(T, NaN))
end

_quantile(::Type{T}, d::Beta, q::Number) where {T} = _invert(T, d, q)

"""
    _invert(T, d::Beta, q)

Solve ``I_x(\\alpha, \\beta) = q`` for `x`.

Bisection first, then Newton steps that fall back to the bracket midpoint whenever they
step outside it. Both loops run a fixed number of times, so the result is a smooth
function of the shapes and carries their derivative.
"""
function _invert(::Type{T}, d::Beta, q::Number) where {T}
    #=
      Tracked numbers leave `T` under arithmetic, so let promotion name the type the
      loops below work in rather than assuming `one(T) / 2` is still a `T`.
    =#
    α, β, p, x = promote(convert(T, d.α), convert(T, d.β), convert(T, q), one(T) / 2)
    W = typeof(x)
    lo, hi = zero(W), one(W)
    # Bracket the root, stopping short of the working resolution.
    for _ in 1:bisections(W)
        below = incbeta(α, β, x) < p
        lo = pick(below, x, lo)
        hi = pick(below, hi, x)
        x = (lo + hi) / 2
    end
    # Each Newton step doubles the correct digits, so four reach `BigFloat` precision.
    for _ in 1:4
        c = incbeta(α, β, x)
        below = c < p
        lo = pick(below, x, lo)
        hi = pick(below, hi, x)
        # A zero density makes the step infinite, which the bracket test then rejects.
        candidate = x - (c - p) / exp(logdensityof(d, x))
        # Accept a candidate that lands on an endpoint: once Newton has converged, the
        # root is an endpoint, and rejecting it there would bisect away from the answer.
        inside = (candidate >= lo) & (candidate <= hi)
        x = pick(inside, candidate, (lo + hi) / 2)
    end
    valid = checkparams(d) & (p >= zero(W)) & (p <= one(W))
    return pick(valid, x, convert(W, NaN))
end

function Base.show(io::IO, d::Beta)
    return print(io, "Beta(α=", d.α, ", β=", d.β, ")")
end
