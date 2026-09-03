"""
    Exponential(θ)
    Exponential()

The exponential measure on ``[0, \\infty)`` with scale `θ`. Its density is

```math
p(x) = \\frac{1}{\\theta} \\exp(-x/\\theta)
```

`θ` is the mean, matching Distributions.jl. `Exponential()` creates the unit-scale
measure using `Float64`.

# Arguments

  - `θ::Number`: the scale.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Exponential(-1.0))               # false
isnan(logdensityof(Exponential(-1.0), 1.0))  # true
```
"""
struct Exponential{R<:Number} <: ContinuousUnivariateMeasure
    θ::R
end

Exponential() = Exponential(1.0)

Base.eltype(::Type{Exponential{R}}) where {R} = float(R)

checkparams(d::Exponential) = isfinite(d.θ) & (d.θ > zero(d.θ))

support(::Exponential) = NonNegativeReals()

"""
    sval(d, x)

Return the scaled value ``x/\\theta`` for a scale-parameterized measure.
"""
@inline sval(d::Exponential, x::Number) = x / d.θ

@inline function DensityInterface.logdensityof(d::Exponential, x::Number)
    s = sval(d, x)
    θ = oftype(s, d.θ)
    # Convert exact values to a float before returning `-Inf`.
    return select(x >= zero(x), () -> -(s + logt(θ)), () -> oftype(float(s), -Inf))
end

# Apply `θ` after drawing noise so automatic differentiation can follow it.
@inline function Base.rand(rng::AbstractRNG, d::Exponential)
    e = -log(rand(rng, noisetype(d)))
    return d.θ * e
end

Statistics.mean(d::Exponential) = d.θ
Statistics.var(d::Exponential) = d.θ^2

# Keep this equal to `sqrt(var(d))` even when `θ` is invalid and negative.
Statistics.std(d::Exponential) = abs(d.θ)

function entropy(d::Exponential)
    θ = float(d.θ)
    return one(θ) + logt(θ)
end

function cdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> -expm1(-s), () -> zero(s))
end

function ccdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> exp(-s), () -> one(s))
end

function logcdf(d::Exponential, x::Number)
    s = sval(d, x)
    # Convert exact values to a float before returning `-Inf`.
    return select(x >= zero(x), () -> log1mexpt(-s), () -> oftype(float(s), -Inf))
end

# This direct expression is accurate for every `x`.
function logccdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> -s, () -> zero(s))
end

# `log1pt` returns `NaN` instead of throwing when `p` is outside `[0, 1]`.
Statistics.quantile(d::Exponential, p::Number) = -d.θ * log1pt(-p)

function Base.show(io::IO, d::Exponential)
    return print(io, "Exponential(θ=", d.θ, ")")
end
