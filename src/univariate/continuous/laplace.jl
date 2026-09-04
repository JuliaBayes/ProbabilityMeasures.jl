"""
    Laplace(μ, b)
    Laplace()

The Laplace (double exponential) measure on ``\\mathbb{R}`` with location `μ` and
scale `b`. Its density is

```math
p(x) = \\frac{1}{2b} \\exp\\!\\left(-\\frac{|x-\\mu|}{b}\\right)
```

`Laplace()` creates the standard Laplace measure using `Float64` values.

# Arguments

  - `μ::Number`: the location, which is also the mean and the median.
  - `b::Number`: the positive scale.

The density has a sharp point at `x = μ`. Its value is continuous there, but the
slope of its log changes from `1/b` to `-1/b`.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Laplace(0.0, -1.0))               # false
isnan(logdensityof(Laplace(0.0, -1.0), 0.0))  # true
```
"""
struct Laplace{M<:Number,B<:Number} <: ContinuousUnivariateMeasure
    μ::M
    b::B
end

Laplace() = Laplace(0.0, 1.0)

Base.eltype(::Type{Laplace{M,B}}) where {M,B} = float(promote_type(M, B))

checkparams(d::Laplace) = isfinite(d.μ) & isfinite(d.b) & (d.b > zero(d.b))

support(::Laplace) = RealLine()

"""
    zval(d::Laplace, x)

Return the standardized value ``(x - \\mu)/b``.
"""
@inline zval(d::Laplace, x::Number) = (x - d.μ) / d.b

"""
    xval(d::Laplace, z)

Convert `z` back to the original scale: ``\\mu + b z``.
"""
@inline xval(d::Laplace, z::Number) = muladd(d.b, z, d.μ)

@inline function DensityInterface.logdensityof(d::Laplace, x::Number)
    z = zval(d, x)
    # Match `b` to `z` so integer parameters do not reduce `BigFloat` precision.
    b = oftype(z, d.b)
    return -abs(z) - logt(2 * b)
end

#=
  The difference of two unit exponential samples follows Laplace(0, 1). Applying `μ`
  and `b` afterward lets automatic differentiation pass through the sample. `log1p(-u)`
  rather than `log(u)`, so an exact zero draw gives a finite sample, not `Inf` or `NaN`.
=#
@inline function Base.rand(rng::AbstractRNG, d::Laplace)
    T = noisetype(d)
    e₁ = -log1p(-rand(rng, T))
    e₂ = -log1p(-rand(rng, T))
    return xval(d, e₁ - e₂)
end

Statistics.mean(d::Laplace) = d.μ
Statistics.median(d::Laplace) = d.μ
Statistics.var(d::Laplace) = 2 * d.b^2

# Keep this equal to `sqrt(var(d))` even when `b` is invalid and negative.
Statistics.std(d::Laplace) = sqrt2 * abs(d.b)

function entropy(d::Laplace)
    b = float(d.b)
    return one(b) + logt(2 * b)
end

# Half of the probability lies on each side of `μ`.
function cdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> exp(z) / 2, () -> 1 - exp(-z) / 2)
end

function ccdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> exp(-z) / 2, () -> 1 - exp(z) / 2)
end

# Compute the small tail directly so its log stays finite. Some tools evaluate both
# branches; `log1pt` keeps the unused branch from throwing when its input is below -1.
function logcdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> z - logtwo, () -> log1pt(-exp(-z) / 2))
end

function logccdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> -z - logtwo, () -> log1pt(-exp(z) / 2))
end

# Use `p` in the lower half and `1 - p` in the upper half to preserve small
# probabilities. Outside `[0, 1]`, `logt` returns `NaN` instead of throwing.
function Statistics.quantile(d::Laplace, p::Number)
    half = one(p) / 2
    return select(p < half, () -> xval(d, logt(2 * p)), () -> xval(d, -logt(2 * (1 - p))))
end

function Base.show(io::IO, d::Laplace)
    return print(io, "Laplace(μ=", d.μ, ", b=", d.b, ")")
end
