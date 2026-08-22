"""
    Cauchy(μ, σ)
    Cauchy()

The Cauchy measure on ``\\mathbb{R}`` with location `μ` and positive scale `σ`.

`Cauchy()` creates the standard Cauchy measure using `Float64` values. Its mean,
variance, and standard deviation are undefined and return `NaN`.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.
"""
struct Cauchy{M<:Number,S<:Number} <: ContinuousUnivariateMeasure
    μ::M
    σ::S
end

Cauchy() = Cauchy(0.0, 1.0)

Base.eltype(::Type{Cauchy{M,S}}) where {M,S} = float(promote_type(M, S))

checkparams(d::Cauchy) = isfinite(d.μ) & isfinite(d.σ) & (d.σ > zero(d.σ))

support(::Cauchy) = RealLine()

@inline zval(d::Cauchy, x::Number) = (x - d.μ) / d.σ

@inline xval(d::Cauchy, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Cauchy, x::Number)
    z = zval(d, x)
    σ = oftype(z, d.σ)
    return -logt(σ) - logπ - 2 * log(hypot(one(z), z))
end

@inline function Base.rand(rng::AbstractRNG, d::Cauchy)
    u = rand(rng, noisetype(d))
    return xval(d, -cospi(u) / sinpi(u))
end

Statistics.mean(d::Cauchy) = oftype(zero(eltype(d)), NaN)
Statistics.var(d::Cauchy) = oftype(zero(eltype(d)), NaN)
Statistics.median(d::Cauchy) = d.μ

function entropy(d::Cauchy)
    σ = float(d.σ)
    return logt(σ) + logπ + logtwo + logtwo
end

function cdf(d::Cauchy, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> atan(-inv(z)) / π, () -> one(z) / 2 + atan(z) / π)
end

function ccdf(d::Cauchy, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> atan(inv(z)) / π, () -> one(z) / 2 - atan(z) / π)
end

function logcdf(d::Cauchy, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> logt(cdf(d, x)), () -> log1p(-ccdf(d, x)))
end

function logccdf(d::Cauchy, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> logt(ccdf(d, x)), () -> log1p(-cdf(d, x)))
end

function Statistics.quantile(d::Cauchy, p::Number)
    valid = (p >= zero(p)) & (p <= one(p))
    q = abs(select(valid, () -> p, () -> one(float(p)) / 2))
    z = -cospi(q) / sinpi(q)
    return select(valid, () -> xval(d, z), () -> oftype(float(z), NaN))
end

function Base.show(io::IO, d::Cauchy)
    return print(io, "Cauchy(μ=", d.μ, ", σ=", d.σ, ")")
end
