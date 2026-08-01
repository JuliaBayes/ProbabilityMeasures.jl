"""
    Normal(μ, σ)
    Normal()

The normal (Gaussian) measure on ``\\mathbb{R}`` with mean `μ` and standard
deviation `σ`, with density

```math
p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

with respect to Lebesgue measure.

# Parameters are not promoted

`μ` and `σ` keep their own types. Nothing is converted at construction, so an AD
dual entering through `μ` leaves `σ` alone:

```julia
typeof(Normal(ForwardDiff.Dual(0.0, 1.0), 1.0))
# Normal{ForwardDiff.Dual{Nothing, Float64, 1}, Float64}
```

Integer parameters do not force `Float64` either -- the precision follows the
argument, not the parameters:

```julia
logdensityof(Normal(0, 1), 1.0f0) isa Float32     # true
logdensityof(Normal(0, 1), big"1.0") isa BigFloat # true
```

# Construction never validates

`Normal(0.0, -1.0)` builds without complaint; see [`checkparams`](@ref) for why.
The density is total, so an invalid scale surfaces as `NaN` rather than a throw:

```julia
checkparams(Normal(0.0, -1.0))               # false
isnan(logdensityof(Normal(0.0, -1.0), 0.0))  # true
```
"""
struct Normal{M<:Real,S<:Real} <: ContinuousUnivariateMeasure
    μ::M
    σ::S
end

# There is deliberately no outer constructor here. Julia's auto-generated
# `Normal(μ::M, σ::S) where {M,S}` already keeps both types intact -- it is
# `Normal(μ, σ) = Normal(promote(μ, σ)...)` that takes extra code to write, not the
# generic version. Argument checking is likewise absent by design; see `checkparams`.

# `Float64` here is a *default*, not a constraint -- it is what you get when you
# decline to say anything at all. Write `Normal(0.0f0, 1.0f0)` for single precision.
Normal() = Normal(0.0, 1.0)

Base.eltype(::Type{Normal{M,S}}) where {M,S} = float(promote_type(M, S))

checkparams(d::Normal) = isfinite(d.μ) && d.σ > zero(d.σ)

support(::Normal) = RealLine()

"""
    zval(d::Normal, x)

The standardized value ``(x - \\mu)/\\sigma``.
"""
@inline zval(d::Normal, x::Real) = (x - d.μ) / d.σ

"""
    xval(d::Normal, z)

The inverse of [`zval`](@ref): ``\\mu + \\sigma z``.
"""
@inline xval(d::Normal, z::Real) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Real)
    z = zval(d, x)
    # `z` already carries the fully promoted type, so converting `σ` to it before
    # taking the log is what keeps precision correct for exact parameter types: with
    # `μ::BigFloat, σ::Int`, `log(σ)` alone would compute at `Float64` precision and
    # quietly cap the result. `log2π` is an `Irrational` and adopts `z`'s type (and,
    # for `BigFloat`, its current precision) on its own.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - logt(σ)
end

# Reparameterized by construction: the randomness is drawn in a plain float type and
# the parameters enter through arithmetic, so pathwise (VI) gradients fall out with
# no custom AD rules.
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, noisetype(d)))
end

# --- Moments --------------------------------------------------------------------

Statistics.mean(d::Normal) = d.μ
Statistics.var(d::Normal) = d.σ^2
# `abs` rather than a bare `d.σ`: for an invalid negative scale the two would
# disagree with `var`, and anything using `std` as a proposal width or a tolerance
# would get a negative number. `abs` is exact, type-preserving, and identical to
# `sqrt(var(d))` for every valid scale.
Statistics.std(d::Normal) = abs(d.σ)
Statistics.median(d::Normal) = d.μ
mode(d::Normal) = d.μ
skewness(d::Normal) = zero(eltype(d))
kurtosis(d::Normal) = zero(eltype(d))

function entropy(d::Normal)
    σ = float(d.σ)
    # `one(σ)` rather than the literal `1`: `log2π + 1` would evaluate the
    # `Irrational` at `Float64` and cap a `BigFloat` result at 53 bits.
    return (log2π + one(σ)) / 2 + logt(σ)
end

mgf(d::Normal, t::Real) = exp(t * d.μ + (t * d.σ)^2 / 2)
cf(d::Normal, t::Real) = exp(im * t * d.μ - (t * d.σ)^2 / 2)

# --- Distribution function ------------------------------------------------------

cdf(d::Normal, x::Real) = erfc(-zval(d, x) * invsqrt2) / 2
ccdf(d::Normal, x::Real) = erfc(zval(d, x) * invsqrt2) / 2

# Both tails need care, and they need *different* care.
#
# Far below the mean, `cdf` underflows to zero and `log(cdf(...))` returns `-Inf`
# where the true value is merely large and negative; `logerfc` computes it directly.
# Far above the mean the opposite happens: `cdf` rounds to exactly one and the log
# collapses to `0.0`, destroying every significant digit of a value that is small
# but nonzero. There `log1p(-ccdf)` keeps full relative accuracy.
#
# A PPL walks into both regimes constantly -- censored likelihoods and truncations
# evaluate `logcdf` precisely where the mass runs out.
function logcdf(d::Normal, x::Real)
    z = zval(d, x)
    return z < zero(z) ? logerfc(-z * invsqrt2) - logtwo : log1p(-erfc(z * invsqrt2) / 2)
end

function logccdf(d::Normal, x::Real)
    z = zval(d, x)
    return z > zero(z) ? logerfc(z * invsqrt2) - logtwo : log1p(-erfc(-z * invsqrt2) / 2)
end

# The parenthesisation is load-bearing. `-sqrt2 * erfcinv(...)` parses as
# `(-sqrt2) * erfcinv(...)`, and Base defines `-(x::AbstractIrrational) = -Float64(x)`
# -- so unary minus materializes √2 at Float64 *before* it ever meets the argument,
# throwing away the Irrational promotion the rest of this file depends on. Negating
# last keeps `sqrt2` a binary operand, so it adopts the argument's type and (for
# BigFloat) its precision.
Statistics.quantile(d::Normal, p::Real) = xval(d, -(sqrt2 * erfcinv(2 * p)))

function Base.show(io::IO, d::Normal)
    return print(io, "Normal(μ=", d.μ, ", σ=", d.σ, ")")
end
