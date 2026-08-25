"""
    Poisson(λ)

The Poisson measure on ``\\{0, 1, 2, \\ldots\\}`` with rate `λ` and probability mass
function

```math
P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}.
```

`λ` is both the mean and variance. It must be finite and non-negative. The constructor
does not validate it; use [`validateparams`](@ref) for user input.

Density results follow Julia's promotion rules for `λ` and the evaluation point.
Samples use `float(typeof(λ))`.

`logdensityof` is constant time, and so are `cdf`, `ccdf`, `quantile`, and `rand` for
plain floating-point arguments, which use the regularized incomplete gamma. The wrapped
numbers differentiation and tracing tools substitute are `Real` but not `AbstractFloat`,
so they sum through [`horizon(d)`](@ref) at `O(λ)` and cannot run in traced or
device-side code. `entropy` has no closed form and always sums. All five return `NaN`
when `λ` is invalid or no usable `Int` horizon exists.
"""
struct Poisson{L<:Number} <: DiscreteUnivariateMeasure
    λ::L
end

Base.eltype(::Type{Poisson{L}}) where {L} = float(L)

checkparams(d::Poisson) = isfinite(d.λ) & (d.λ >= zero(d.λ))

support(::Poisson) = NonNegativeIntegers()

"""
    horizon(d::Poisson)

The largest outcome a sum over the support of `d` includes.

The horizon is ``\\lceil \\lambda + 20\\sqrt{\\lambda} + 60 \\rceil``. The omitted tail is
smaller than the rounding error at the default `BigFloat` precision. `horizon` returns
zero if the bound is non-positive, non-finite, or too large for `Int`.
"""
@inline function horizon(d::Poisson)
    λ = float(d.λ)
    n = λ + 20 * sqrt(max(λ, zero(λ))) + 60
    # Guard `ceil(Int, n)` against overflow and non-finite values.
    return ((n > zero(n)) & (n <= typemax(Int))) ? ceil(Int, n) : 0
end

@inline function DensityInterface.logdensityof(d::Poisson, x::Number)
    T = masstype(d, x)
    λ, k = convert(T, d.λ), convert(T, x)
    # Define `0 * log(0)` as zero.
    a = select(k == zero(T), () -> zero(T), () -> k * logt(λ))
    # `loggamma` rejects negative integers before the support check runs.
    g = loggamma(max(k, zero(T)) + one(T))
    # If `log(k!)` overflows, the mass is zero; avoid `Inf - Inf`.
    return select(insupport(d, x) & isfinite(g), () -> a - λ - g, () -> convert(T, -Inf))
end

# CDF inversion avoids the `exp(-λ)` underflow in product-of-uniforms sampling. Draws
# are plain floats, so this reaches the closed-form quantile.
@inline function Base.rand(rng::AbstractRNG, d::Poisson)
    return quantile(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Poisson) = d.λ
Statistics.var(d::Poisson) = d.λ

"""
    usable(d::Poisson)

Whether the distribution functions of `d` can return a number.

They cannot when `λ` fails [`checkparams`](@ref) or [`horizon`](@ref) is zero. Both give
`NaN`, from either path.
"""
@inline usable(d::Poisson) = checkparams(d) & (horizon(d) > 0)

# Entropy has no closed form, so it always sums.
function entropy(d::Poisson)
    T = eltype(d)
    h = zero(T)
    for k in 0:horizon(d)
        logp = logdensityof(d, convert(T, k))
        # Treat `0 * -Inf` as zero.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return select(usable(d), () -> h, () -> convert(T, NaN))
end

#=
  `masstype` picks the path. Plain floats, `BigFloat` included, take the closed forms;
  everything else takes the sums. The wrapped numbers differentiation and tracing tools
  substitute are `Real` but not `AbstractFloat`, and `gamma_inc` has no method for them.
=#
cdf(d::Poisson, x::Number) = _cdf(masstype(d, x), d, x)
ccdf(d::Poisson, x::Number) = _ccdf(masstype(d, x), d, x)
Statistics.quantile(d::Poisson, q::Number) = _quantile(masstype(d, q), d, q)

# `P(X ≤ k) = Q(k+1, λ)` and `P(X > k) = P(k+1, λ)`. One call returns both tails, so
# neither is left to a subtraction from one.
@inline function _tails(::Type{T}, d::Poisson, x::Number) where {T<:AbstractFloat}
    nan = convert(T, NaN)
    usable(d) || return (nan, nan)
    k = floor(convert(T, x))
    isnan(k) && return (nan, nan)
    # `gamma_inc` needs a positive first argument, and the tails are known below zero.
    k < zero(T) && return (zero(T), one(T))
    lower, upper = gamma_inc(k + one(T), convert(T, d.λ))
    return (upper, lower)
end

_cdf(::Type{T}, d::Poisson, x::Number) where {T<:AbstractFloat} = _tails(T, d, x)[1]
_ccdf(::Type{T}, d::Poisson, x::Number) where {T<:AbstractFloat} = _tails(T, d, x)[2]

function _quantile(::Type{T}, d::Poisson, q::Number) where {T<:AbstractFloat}
    usable(d) || return convert(T, NaN)
    p = convert(T, q)
    hi = convert(T, horizon(d))
    p >= one(T) && return hi
    # `NaN` lands here too, and on the first outcome, which is where the sum leaves it.
    p > zero(T) || return zero(T)
    λ = convert(T, d.λ)
    # A normal approximation with a continuity correction starts within a step or two.
    z = convert(T, -sqrt2 * erfcinvt(2 * p))
    k = clamp(floor(λ + z * sqrt(λ) - one(T) / 2), zero(T), hi)
    # Walk to the first outcome whose CDF reaches `p`; only one loop runs. Searching
    # with `_cdf` itself is what keeps `quantile(d, cdf(d, k))` exact.
    while (k > zero(T)) && (_cdf(T, d, k - one(T)) >= p)
        k -= one(T)
    end
    while (k < hi) && (_cdf(T, d, k) < p)
        k += one(T)
    end
    return k
end

# Sum the lower tail directly and clamp roundoff above one.
function _cdf(::Type{T}, d::Poisson, x::Number) where {T}
    c = zero(T)
    for k in 0:horizon(d)
        c += select(k <= x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return select(usable(d), () -> min(c, one(T)), () -> convert(T, NaN))
end

# Beyond the horizon this returns zero rather than the true remaining mass.
function _ccdf(::Type{T}, d::Poisson, x::Number) where {T}
    c = zero(T)
    for k in 0:horizon(d)
        c += select(k > x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return select(usable(d), () -> min(c, one(T)), () -> convert(T, NaN))
end

# Complete the loop to keep control flow independent of `q`. Matching `_cdf`'s summation
# order preserves `quantile(d, cdf(d, k)) == k` until the CDF rounds to one.
function _quantile(::Type{T}, d::Poisson, q::Number) where {T}
    kmax = horizon(d)
    hi = convert(T, kmax)
    total = zero(T)
    i = zero(T)
    for k in 0:kmax
        total += exp(logdensityof(d, convert(T, k)))
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # Copy the loop-assigned value to avoid boxing the closure capture.
    below = i
    # The horizon stands in for the infinite quantile at `q >= 1`.
    value = select(q >= one(T), () -> hi, () -> min(below, hi))
    return select(usable(d), () -> value, () -> convert(T, NaN))
end

function Base.show(io::IO, d::Poisson)
    return print(io, "Poisson(λ=", d.λ, ")")
end
