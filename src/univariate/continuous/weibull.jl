"""
    Weibull(α, θ)
    Weibull()

The Weibull measure on ``[0, \\infty)`` with shape `α` and scale `θ`. Its density is

```math
p(x) = \\frac{\\alpha}{\\theta}
       \\left(\\frac{x}{\\theta}\\right)^{\\alpha - 1}
       \\exp\\!\\left[-\\left(\\frac{x}{\\theta}\\right)^\\alpha\\right]
```

The parameterization matches Distributions.jl. `Weibull(1, θ)` is `Exponential(θ)`,
and the density at zero follows the shape: infinite for `α < 1`, `1/θ` at `α = 1`,
and zero above.

`Weibull()` creates the unit-shape, unit-scale measure using `Float64` values.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.
"""
struct Weibull{A<:Number,T<:Number} <: ContinuousUnivariateMeasure
    α::A
    θ::T
end

Weibull() = Weibull(1.0, 1.0)

Base.eltype(::Type{Weibull{A,T}}) where {A,T} = float(promote_type(A, T))

function checkparams(d::Weibull)
    return isfinite(d.α) & isfinite(d.θ) & (d.α > zero(d.α)) & (d.θ > zero(d.θ))
end

support(::Weibull) = NonNegativeReals()

@inline sval(d::Weibull, x::Number) = x / d.θ

"""
    tval(d::Weibull, x)

Return the transformed value ``(x/\\theta)^\\alpha``.

The `abs` keeps the power from a domain error when tools that evaluate both `select`
branches reach it with a negative `x`.
"""
@inline function tval(d::Weibull, x::Number)
    s, α = map(float, promote(sval(d, x), d.α))
    return abs(s)^α
end

@inline function DensityInterface.logdensityof(d::Weibull, x::Number)
    scaled, α = map(float, promote(sval(d, x), d.α))
    s = abs(scaled)
    # Match `θ` to `s` so integer parameters do not reduce `BigFloat` precision.
    θ = oftype(s, d.θ)
    # At an infinite `s` the log and power terms cancel to `NaN`; the density is zero.
    value = select(
        isinf(s),
        () -> oftype(s, -Inf),
        () -> logt(α) - logt(θ) + (α - one(α)) * logt(s) - s^α,
    )
    # At `x = 0` the formula settles every shape but `α = 1`, where `0 * log(0)` is `NaN`.
    atzero = select(α == one(α), () -> -logt(θ), () -> value)
    return select(
        x > zero(x),
        () -> value,
        () -> select(x == zero(x), () -> atzero, () -> oftype(s, -Inf)),
    )
end

#=
  A draw is `θ E^(1/α)` for a unit exponential `E`. Applying the parameters after
  drawing noise lets automatic differentiation follow them.
=#
@inline function Base.rand(rng::AbstractRNG, d::Weibull)
    e = -log(rand(rng, noisetype(d)))
    α, θ = promote(d.α, d.θ)
    return θ * e^inv(α)
end

function Statistics.mean(d::Weibull)
    α, θ = map(float, promote(d.α, d.θ))
    return θ * exp(loggamma(one(α) + inv(α)))
end

function Statistics.var(d::Weibull)
    α, θ = map(float, promote(d.α, d.θ))
    g₁ = loggamma(one(α) + inv(α))
    g₂ = loggamma(one(α) + 2 / α)
    return θ^2 * exp(2g₁) * expm1(g₂ - 2g₁)
end

function entropy(d::Weibull)
    α, θ = map(float, promote(d.α, d.θ))
    γ = oftype(α, Base.MathConstants.eulergamma)
    return γ * (one(α) - inv(α)) + logt(θ / α) + one(α)
end

function cdf(d::Weibull, x::Number)
    t = tval(d, x)
    return select(x >= zero(x), () -> -expm1(-t), () -> zero(t))
end

function ccdf(d::Weibull, x::Number)
    t = tval(d, x)
    return select(x >= zero(x), () -> exp(-t), () -> one(t))
end

function logcdf(d::Weibull, x::Number)
    t = tval(d, x)
    return select(x >= zero(x), () -> log1mexpt(-t), () -> oftype(t, -Inf))
end

# This direct expression is accurate for every `x`.
function logccdf(d::Weibull, x::Number)
    t = tval(d, x)
    return select(x >= zero(x), () -> -t, () -> zero(t))
end

function Statistics.quantile(d::Weibull, p::Number)
    q, α, θ = map(float, promote(p, d.α, d.θ))
    #=
      `log1pt` returns `NaN` when `q` exceeds one, so only a negative `q` needs the
      guard. The `abs` keeps the power from a domain error when tools that evaluate
      both branches reach it with a negative `q`.
    =#
    value = θ * abs(log1pt(-q))^inv(α)
    return select(q >= zero(q), () -> value, () -> oftype(q, NaN))
end

function Base.show(io::IO, d::Weibull)
    return print(io, "Weibull(α=", d.α, ", θ=", d.θ, ")")
end
