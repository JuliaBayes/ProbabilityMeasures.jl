"""
    Weibull(α, θ)
    Weibull()

The Weibull measure on ``[0, \\infty)`` with shape `α` and scale `θ`. Its density is

```math
p(x) = \\frac{\\alpha}{\\theta}
       \\left(\\frac{x}{\\theta}\\right)^{\\alpha - 1}
       \\exp\\!\\left[-\\left(\\frac{x}{\\theta}\\right)^\\alpha\\right]
```

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

@inline function DensityInterface.logdensityof(d::Weibull, x::Number)
    scaled, shape = promote(sval(d, x), d.α)
    s = abs(float(scaled))
    α, θ = oftype(s, shape), oftype(s, d.θ)
    value = logt(α) - logt(θ) + (α - one(α)) * logt(s) - s^α
    atzero = select(
        α < one(α),
        () -> oftype(s, Inf),
        () -> select(α == one(α), () -> -logt(θ), () -> oftype(s, -Inf)),
    )
    return select(
        x > zero(x),
        () -> value,
        () -> select(x == zero(x), () -> atzero, () -> oftype(s, -Inf)),
    )
end

@inline function Base.rand(rng::AbstractRNG, d::Weibull)
    return quantile(d, rand(rng, noisetype(d)))
end

function Statistics.mean(d::Weibull)
    α, θ = promote(d.α, d.θ)
    α = float(α)
    return oftype(α, θ) * exp(loggamma(one(α) + inv(α)))
end

function Statistics.var(d::Weibull)
    α, θ = promote(d.α, d.θ)
    α, θ = float(α), oftype(float(α), θ)
    g₁ = loggamma(one(α) + inv(α))
    g₂ = loggamma(one(α) + 2 / α)
    return θ^2 * exp(2g₁) * expm1(g₂ - 2g₁)
end

function entropy(d::Weibull)
    α, θ = promote(d.α, d.θ)
    α, θ = float(α), oftype(float(α), θ)
    γ = oftype(α, Base.MathConstants.eulergamma)
    return γ * (one(α) - inv(α)) + logt(θ / α) + one(α)
end

function cdf(d::Weibull, x::Number)
    s, α = promote(sval(d, x), d.α)
    t = abs(float(s))^oftype(float(s), α)
    return select(x >= zero(x), () -> -expm1(-t), () -> zero(t))
end

function ccdf(d::Weibull, x::Number)
    s, α = promote(sval(d, x), d.α)
    t = abs(float(s))^oftype(float(s), α)
    return select(x >= zero(x), () -> exp(-t), () -> one(t))
end

function logcdf(d::Weibull, x::Number)
    s, α = promote(sval(d, x), d.α)
    t = abs(float(s))^oftype(float(s), α)
    return select(x >= zero(x), () -> log1mexpt(-t), () -> oftype(t, -Inf))
end

function logccdf(d::Weibull, x::Number)
    s, α = promote(sval(d, x), d.α)
    t = abs(float(s))^oftype(float(s), α)
    return select(x >= zero(x), () -> -t, () -> zero(t))
end

function Statistics.quantile(d::Weibull, p::Number)
    probability, shape, scale = promote(p, d.α, d.θ)
    q = float(probability)
    α, θ = oftype(q, shape), oftype(q, scale)
    value = θ * abs(-log1pt(-q))^inv(α)
    valid = (q >= zero(q)) & (q <= one(q))
    return select(valid, () -> value, () -> oftype(q, NaN))
end

function Base.show(io::IO, d::Weibull)
    return print(io, "Weibull(α=", d.α, ", θ=", d.θ, ")")
end
