"""
    TDist(ν, μ, σ)
    TDist(ν)

Student's t measure on ``\\mathbb{R}`` with `ν` degrees of freedom, location `μ` and
positive scale `σ`. Its density is

```math
p(x) = \\frac{\\Gamma\\!\\left(\\frac{\\nu+1}{2}\\right)}
             {\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\nu\\pi}\\,\\sigma}
       \\left(1 + \\frac{1}{\\nu}\\left(\\frac{x-\\mu}{\\sigma}\\right)^{2}\\right)
       ^{-\\frac{\\nu+1}{2}}.
```

`TDist(ν)` places it at zero with unit scale, using `Float64` for both. `σ` is a scale,
not a standard deviation: the variance is ``\\sigma^2 \\nu / (\\nu - 2)``.

Only moments below `ν` exist. The mean is `NaN` for ``\\nu \\le 1``, and the variance is
`Inf` for ``1 < \\nu \\le 2`` and `NaN` below that. ``\\nu = 1`` is [`Cauchy`](@ref),
which has closed forms where this measure iterates.

The distribution functions and `rand` go through the regularized incomplete beta, so
they need a plain floating-point type; the wrapped numbers differentiation and tracing
tools substitute have no method. A draw still carries derivatives in `μ` and `σ`, which
it picks up after the inversion, but not in `ν`. `logdensityof` and `entropy` are closed
forms and differentiate in all three. `logcdf` and `logccdf` take the logarithm inside
the incomplete beta and so stay finite far past the point where `cdf` and `ccdf`
underflow.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.
"""
struct TDist{N<:Number,M<:Number,S<:Number} <: ContinuousUnivariateMeasure
    ν::N
    μ::M
    σ::S
end

TDist(ν::Number) = TDist(ν, 0.0, 1.0)

Base.eltype(::Type{TDist{N,M,S}}) where {N,M,S} = float(promote_type(N, M, S))

function checkparams(d::TDist)
    return isfinite(d.ν) & (d.ν > zero(d.ν)) & isfinite(d.μ) & isfinite(d.σ) &
           (d.σ > zero(d.σ))
end

support(::TDist) = RealLine()

@inline zval(d::TDist, x::Number) = (x - d.μ) / d.σ

@inline xval(d::TDist, z::Number) = muladd(d.σ, z, d.μ)

"""
    tlognorm(ν, n=1)

The log normalizing constant of the standard `n`-dimensional t measure,
``\\log \\Gamma(\\frac{\\nu+n}{2}) - \\log \\Gamma(\\frac{\\nu}{2}) -
\\frac{n}{2}\\log(\\nu\\pi)``.

[`MvTDist`](@ref) uses the same constant with `n` its dimension.
"""
@inline function tlognorm(ν::Number, n::Integer=1)
    half = (ν + n) / 2
    return loggammat(half) - loggammat(ν / 2) - n * (logt(ν) + oftype(float(ν), logπ)) / 2
end

@inline function DensityInterface.logdensityof(d::TDist, x::Number)
    z = zval(d, x)
    T = float(promote_type(typeof(z), typeof(d.ν)))
    ν = convert(T, d.ν)
    # `hypot` keeps the squared standardized value from overflowing.
    u = convert(T, z) / sqrtt(ν)
    return tlognorm(ν) - logt(convert(T, d.σ)) - (ν + one(ν)) * log(hypot(one(u), u))
end

# CDF inversion is the only exact t sampler that does not reject draws.
@inline function Base.rand(rng::AbstractRNG, d::TDist)
    return quantile(d, rand(rng, noisetype(d)))
end

function Statistics.mean(d::TDist)
    T = eltype(d)
    ν = convert(T, d.ν)
    return select(ν > one(T), () -> convert(T, d.μ), () -> convert(T, NaN))
end

function Statistics.var(d::TDist)
    T = eltype(d)
    ν, σ = convert(T, d.ν), convert(T, d.σ)
    # Between one and two degrees of freedom the variance diverges rather than vanishes.
    heavy = select(ν > one(T), () -> convert(T, Inf), () -> convert(T, NaN))
    return select(ν > 2 * one(T), () -> σ^2 * ν / (ν - 2), () -> heavy)
end

Statistics.median(d::TDist) = d.μ

function entropy(d::TDist)
    T = eltype(d)
    ν = convert(T, d.ν)
    half = (ν + one(T)) / 2
    return half * (digamma(half) - digamma(ν / 2)) - tlognorm(ν) + logt(convert(T, d.σ))
end

"""
    tsplit(d::TDist, x) -> (usable, z, ν, y, yc)

The arguments of the incomplete beta behind the distribution functions of `d`.

``I_y(\\nu/2, 1/2) = P(|T| > |z|)`` with ``y = \\nu/(\\nu + z^2)``. Both `y` and its
complement are returned so neither loses its leading digits, and squaring whichever of
``|z|/\\sqrt{\\nu}`` and its reciprocal is smaller keeps a large `z` from overflowing.
`usable` is false when the parameters or `x` leave nothing to compute.
"""
@inline function tsplit(d::TDist, x::Number)
    T = float(promote_type(typeof(zval(d, x)), typeof(d.ν)))
    z = convert(T, zval(d, x))
    ν = convert(T, d.ν)
    a = abs(z) / sqrtt(ν)
    r = min(a, inv(a))^2
    near, far = r / (one(T) + r), inv(one(T) + r)
    y, yc = a <= one(T) ? (far, near) : (near, far)
    return (checkparams(d) & !isnan(z), z, ν, y, yc)
end

# Halving the incomplete beta gives each tail directly rather than as a subtraction
# from one.
function ttails(d::TDist, x::Number)
    usable, z, ν, y, yc = tsplit(d, x)
    T = typeof(z)
    usable || return (convert(T, NaN), convert(T, NaN))
    p, q = betainc(ν / 2, one(T) / 2, y, yc)
    lower, upper = p / 2, (one(T) + q) / 2
    return z < zero(T) ? (lower, upper) : (upper, lower)
end

"""
    tlogtail(d::TDist, x)

The logarithm of the tail of `d` beyond `x` on the side `x` falls, which stays finite
where the tail itself underflows.
"""
function tlogtail(d::TDist, x::Number)
    usable, z, ν, y, yc = tsplit(d, x)
    T = typeof(z)
    usable || return convert(T, NaN)
    return logbetainc(ν / 2, one(T) / 2, y, yc) - logtwo
end

cdf(d::TDist, x::Number) = ttails(d, x)[1]
ccdf(d::TDist, x::Number) = ttails(d, x)[2]

function logcdf(d::TDist, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> tlogtail(d, x), () -> log1pt(-ccdf(d, x)))
end

function logccdf(d::TDist, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> tlogtail(d, x), () -> log1pt(-cdf(d, x)))
end

"""
    tquantile(ν, p)

The `p`-quantile of the standard t measure with `ν` degrees of freedom.

Only `ν` and `p` reach the incomplete-beta inverse here, which is what lets a draw carry
derivatives in the location and scale [`quantile`](@ref) applies afterwards.
"""
function tquantile(ν::Number, p::Number)
    T = float(promote_type(typeof(ν), typeof(p)))
    q, dof = convert(T, p), convert(T, ν)
    valid = (q >= zero(T)) & (q <= one(T)) & isfinite(dof) & (dof > zero(T))
    valid || return convert(T, NaN)
    #=
      `2 min(q, 1-q)` is the two-sided tail probability solved for, and `|2q - 1|` its
      complement. Both are formed directly so neither loses its leading digits.
    =#
    tail = 2 * min(q, one(T) - q)
    y, yc = betaincinv(dof / 2, one(T) / 2, tail, abs(2 * q - one(T)))
    z = sqrt(dof * yc / y)
    return q < one(T) / 2 ? -z : z
end

function Statistics.quantile(d::TDist, p::Number)
    x = xval(d, tquantile(d.ν, p))
    return select(checkparams(d), () -> x, () -> oftype(x, NaN))
end

function Base.show(io::IO, d::TDist)
    return print(io, "TDist(ν=", d.ν, ", μ=", d.μ, ", σ=", d.σ, ")")
end
