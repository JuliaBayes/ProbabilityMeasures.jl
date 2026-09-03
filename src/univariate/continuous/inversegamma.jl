"""
    InverseGamma(α, θ)
    InverseGamma(α)

The inverse-gamma measure on ``(0, \\infty)`` with shape `α` and scale `θ`. If ``X``
follows `InverseGamma(α, θ)`, then ``1/X`` follows `Gamma(α, 1/θ)`. Its density is

```math
p(x) = \\frac{\\theta^{\\alpha}}{\\Gamma(\\alpha)}\\, x^{-\\alpha - 1} e^{-\\theta/x}
```

The mean is ``\\theta/(\\alpha - 1)`` for ``\\alpha > 1`` and infinite otherwise; the
variance needs ``\\alpha > 2``. `InverseGamma(α)` sets the scale to one.

# Arguments

  - `α::Number`: the shape.
  - `θ::Number`: the scale.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

`logdensityof` is closed form, so it broadcasts on device arrays and traces. `cdf`,
`ccdf`, `logcdf`, `logccdf`, `quantile`, `median` and `entropy` come from the incomplete
gamma integrals and iterate until their terms stop changing the result, which rules out
traced and device-side evaluation; see [`loggammap`](@ref).

The distribution functions read the *upper* incomplete gamma integral at ``\\theta/x``,
and `quantile` inverts that same tail, so a small probability never has to be recovered
from `1 - p`. Sampling inverts a [`Gamma`](@ref) draw and inherits its derivative with
respect to the parameters.
"""
struct InverseGamma{A<:Number,T<:Number} <: ContinuousUnivariateMeasure
    α::A
    θ::T
end

InverseGamma(α::Number) = InverseGamma(α, one(α))

Base.eltype(::Type{InverseGamma{A,T}}) where {A,T} = float(promote_type(A, T))

function checkparams(d::InverseGamma)
    return isfinite(d.α) & (d.α > zero(d.α)) & isfinite(d.θ) & (d.θ > zero(d.θ))
end

support(::InverseGamma) = PositiveReals()

@inline function DensityInterface.logdensityof(d::InverseGamma, x::Number)
    T = valuetype(d, x)
    α, θ, y = convert(T, d.α), convert(T, d.θ), convert(T, x)
    # `loggamma` throws for a non-positive argument, so an invalid shape takes `NaN`.
    lg = select(α > zero(T), () -> loggamma(α), () -> convert(T, NaN))
    v = muladd(α, logt(θ), -lg) - (α + one(T)) * logt(y) - θ / y
    # Convert exact values to a float before returning `-Inf`.
    return select(insupport(d, y), () -> v, () -> convert(T, -Inf))
end

# The scale carries the numeric type of the unit-scale gamma draw, so the quotient lands
# in `eltype(d)` whatever types the two parameters were given separately.
@inline function Base.rand(rng::AbstractRNG, d::InverseGamma)
    return d.θ / rand(rng, Gamma(d.α, one(d.θ)))
end

function Statistics.mean(d::InverseGamma)
    E = eltype(d)
    α, θ = convert(E, d.α), convert(E, d.θ)
    return select(α > one(E), () -> θ / (α - one(E)), () -> convert(E, Inf))
end

function Statistics.var(d::InverseGamma)
    E = eltype(d)
    α, θ = convert(E, d.α), convert(E, d.θ)
    below = (α - one(E))^2 * (α - 2 * one(E))
    return select(α > 2 * one(E), () -> θ^2 / below, () -> convert(E, Inf))
end

function entropy(d::InverseGamma)
    α = float(d.α)
    # `loggamma` and `digamma` throw for a non-positive argument.
    shape = select(
        α > zero(α), () -> loggamma(α) - (one(α) + α) * digamma(α), () -> oftype(α, NaN)
    )
    return α + logt(float(d.θ)) + shape
end

#=
  `P(X ≤ x) = Q(α, θ/x)`, so the lower tail of this measure is the upper tail of the
  incomplete gamma integral, and the two swap places throughout. As with `Gamma`, the
  four functions differ only in the tail they take and in the two values they hold
  outside `(0, ∞)`.
=#
@inline function invgammatail(tail, d::InverseGamma, x::Number, below, above)
    T = valuetype(d, x)
    checkparams(d) || return convert(T, NaN)
    y = convert(T, x)
    isnan(y) && return convert(T, NaN)
    y > zero(T) || return convert(T, below)
    isfinite(y) || return convert(T, above)
    s = convert(T, d.θ) / y
    # An argument small enough to overflow the ratio is below everything the tail can
    # resolve, which is the same answer as an argument at zero.
    isfinite(s) || return convert(T, below)
    return tail(convert(T, d.α), s)
end

cdf(d::InverseGamma, x::Number) = invgammatail((a, s) -> exp(loggammaq(a, s)), d, x, 0, 1)
ccdf(d::InverseGamma, x::Number) = invgammatail((a, s) -> exp(loggammap(a, s)), d, x, 1, 0)
logcdf(d::InverseGamma, x::Number) = invgammatail(loggammaq, d, x, -Inf, 0)
logccdf(d::InverseGamma, x::Number) = invgammatail(loggammap, d, x, 0, -Inf)

function Statistics.quantile(d::InverseGamma, p::Number)
    T = valuetype(d, p)
    checkparams(d) || return convert(T, NaN)
    q = convert(T, p)
    (isnan(q) | (q < zero(T)) | (q > one(T))) && return convert(T, NaN)
    iszero(q) && return zero(T)
    isone(q) && return convert(T, Inf)
    # `p` measures the upper tail of the incomplete gamma integral, so invert that one.
    return convert(T, d.θ) / gammaquantile(convert(T, d.α), q, false)
end

function Base.show(io::IO, d::InverseGamma)
    return print(io, "InverseGamma(α=", d.α, ", θ=", d.θ, ")")
end
