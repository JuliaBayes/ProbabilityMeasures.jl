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

`logdensityof` is constant time. `entropy`, `cdf`, `ccdf`, `quantile`, and `rand` sum
through [`horizon(d)`](@ref), so they cost `O(λ)` and cannot run in traced or
device-side code. They return `NaN` when no usable `Int` horizon exists.
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

# CDF inversion avoids the `exp(-λ)` underflow in product-of-uniforms sampling.
@inline function Base.rand(rng::AbstractRNG, d::Poisson)
    return quantile(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Poisson) = d.λ
Statistics.var(d::Poisson) = d.λ

# A zero horizon means the support cannot be summed safely.
function entropy(d::Poisson)
    T = eltype(d)
    kmax = horizon(d)
    h = zero(T)
    for k in 0:kmax
        logp = logdensityof(d, convert(T, k))
        # Treat `0 * -Inf` as zero.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return select(kmax > 0, () -> h, () -> convert(T, NaN))
end

# Sum the lower tail directly and clamp roundoff above one.
function cdf(d::Poisson, x::Number)
    T = masstype(d, x)
    kmax = horizon(d)
    c = zero(T)
    for k in 0:kmax
        c += select(k <= x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return select(kmax > 0, () -> min(c, one(T)), () -> convert(T, NaN))
end

# Beyond the horizon this returns zero rather than the true remaining mass.
function ccdf(d::Poisson, x::Number)
    T = masstype(d, x)
    kmax = horizon(d)
    c = zero(T)
    for k in 0:kmax
        c += select(k > x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return select(kmax > 0, () -> min(c, one(T)), () -> convert(T, NaN))
end

# Complete the loop to keep control flow independent of `q`. Matching `cdf`'s summation
# order preserves `quantile(d, cdf(d, k)) == k` until the CDF rounds to one.
function Statistics.quantile(d::Poisson, q::Number)
    T = masstype(d, q)
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
    return select(kmax > 0, () -> value, () -> convert(T, NaN))
end

function Base.show(io::IO, d::Poisson)
    return print(io, "Poisson(λ=", d.λ, ")")
end
