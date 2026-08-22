"""
    LogNormal(μ, σ)
    LogNormal()

The log-normal measure on ``(0, \\infty)``. If ``X`` follows `LogNormal(μ, σ)`,
then ``\\log X`` follows `Normal(μ, σ)`.

`LogNormal()` creates the standard log-normal measure using `Float64` values.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.
"""
struct LogNormal{M<:Number,S<:Number} <: ContinuousUnivariateMeasure
    μ::M
    σ::S
end

LogNormal() = LogNormal(0.0, 1.0)

Base.eltype(::Type{LogNormal{M,S}}) where {M,S} = float(promote_type(M, S))

checkparams(d::LogNormal) = isfinite(d.μ) & isfinite(d.σ) & (d.σ > zero(d.σ))

support(::LogNormal) = PositiveReals()

@inline zval(d::LogNormal, x::Number) = (logt(x) - d.μ) / d.σ

@inline xval(d::LogNormal, z::Number) = exp(muladd(d.σ, z, d.μ))

@inline function DensityInterface.logdensityof(d::LogNormal, x::Number)
    lx = logt(x)
    z = (lx - d.μ) / d.σ
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return select(
        x > zero(x), () -> -(z^2 + log2π) / 2 - logt(σ) - lx, () -> oftype(float(z), -Inf)
    )
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline Base.rand(rng::AbstractRNG, d::LogNormal) = xval(d, randn(rng, noisetype(d)))

Statistics.mean(d::LogNormal) = exp(d.μ + d.σ^2 / 2)
Statistics.var(d::LogNormal) = exp(2d.μ + d.σ^2) * expm1(d.σ^2)

function entropy(d::LogNormal)
    σ = float(d.σ)
    return d.μ + logt(σ) + (log2π + one(σ)) / 2
end

function cdf(d::LogNormal, x::Number)
    z = zval(d, x)
    value = erfc(-z * invsqrt2) / 2
    return select(x > zero(x), () -> value, () -> zero(value))
end

function ccdf(d::LogNormal, x::Number)
    z = zval(d, x)
    value = erfc(z * invsqrt2) / 2
    return select(x > zero(x), () -> value, () -> one(value))
end

# Use a direct formula for each tail to avoid underflow and loss of precision.
function logcdf(d::LogNormal, x::Number)
    z = zval(d, x)
    value = select(
        z < zero(z),
        () -> logerfc(-z * invsqrt2) - logtwo,
        () -> log1p(-erfc(z * invsqrt2) / 2),
    )
    return select(x > zero(x), () -> value, () -> oftype(float(value), -Inf))
end

function logccdf(d::LogNormal, x::Number)
    z = zval(d, x)
    value = select(
        z > zero(z),
        () -> logerfc(z * invsqrt2) - logtwo,
        () -> log1p(-erfc(-z * invsqrt2) / 2),
    )
    return select(x > zero(x), () -> value, () -> zero(value))
end

Statistics.quantile(d::LogNormal, p::Number) = xval(d, -(sqrt2 * erfcinvt(2 * p)))

function Base.show(io::IO, d::LogNormal)
    return print(io, "LogNormal(μ=", d.μ, ", σ=", d.σ, ")")
end
