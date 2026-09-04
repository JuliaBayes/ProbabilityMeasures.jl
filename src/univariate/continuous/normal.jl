"""
    Normal(μ, σ)
    Normal()

The normal (Gaussian) measure on ``\\mathbb{R}`` with mean `μ` and standard
deviation `σ`. Its density is

```math
p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

`Normal()` creates the standard normal using `Float64` values.

# Arguments

  - `μ::Number`: the mean.
  - `σ::Number`: the standard deviation.

# Examples

Parameter types are preserved. Integer parameters do not force `Float64`; result
precision follows the value being evaluated:

```julia
logdensityof(Normal(0, 1), 1.0f0) isa Float32     # true
logdensityof(Normal(0, 1), big"1.0") isa BigFloat # true
```

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Normal(0.0, -1.0))               # false
isnan(logdensityof(Normal(0.0, -1.0), 0.0))  # true
```
"""
struct Normal{M<:Number,S<:Number} <: ContinuousUnivariateMeasure
    μ::M
    σ::S
end

Normal() = Normal(0.0, 1.0)

Base.eltype(::Type{Normal{M,S}}) where {M,S} = float(promote_type(M, S))

# Wrapped comparisons may not produce a `Bool`, so do not use `&&`.
checkparams(d::Normal) = isfinite(d.μ) & isfinite(d.σ) & (d.σ > zero(d.σ))

support(::Normal) = RealLine()

"""
    zval(d::Normal, x)

Return the standardized value ``(x - \\mu)/\\sigma``.
"""
@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ

"""
    xval(d::Normal, z)

Convert `z` back to the original scale: ``\\mu + \\sigma z``.
"""
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - logt(σ)
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, noisetype(d)))
end

Statistics.mean(d::Normal) = d.μ
Statistics.var(d::Normal) = d.σ^2

# Keep this equal to `sqrt(var(d))` even when `σ` is invalid and negative.
Statistics.std(d::Normal) = abs(d.σ)

Statistics.median(d::Normal) = d.μ

function entropy(d::Normal)
    σ = float(d.σ)
    # Keep the constant in the same type as `σ`.
    return (log2π + one(σ)) / 2 + logt(σ)
end

cdf(d::Normal, x::Number) = erfc(-zval(d, x) * invsqrt2) / 2
ccdf(d::Normal, x::Number) = erfc(zval(d, x) * invsqrt2) / 2

# Use a direct formula for each tail to avoid underflow and loss of precision.
function logcdf(d::Normal, x::Number)
    z = zval(d, x)
    return select(
        z < zero(z),
        () -> logerfc(-z * invsqrt2) - logtwo,
        () -> log1p(-erfc(z * invsqrt2) / 2),
    )
end

function logccdf(d::Normal, x::Number)
    z = zval(d, x)
    return select(
        z > zero(z),
        () -> logerfc(z * invsqrt2) - logtwo,
        () -> log1p(-erfc(-z * invsqrt2) / 2),
    )
end

# Negate after multiplication so `sqrt2` takes the argument's type. `erfcinvt` returns
# `NaN` instead of throwing for an invalid probability.
Statistics.quantile(d::Normal, p::Number) = xval(d, -(sqrt2 * erfcinvt(2 * p)))

function Base.show(io::IO, d::Normal)
    return print(io, "Normal(μ=", d.μ, ", σ=", d.σ, ")")
end
