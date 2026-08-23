"""
    Poisson(λ)

The Poisson measure on ``\\{0, 1, 2, \\ldots\\}`` with rate `λ`, with probability mass
function

```math
P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}.
```

# Arguments

  - `λ::Number`: the rate, which is both the mean and the variance.

The result type follows `λ` and the value being evaluated. Samples use
`float(typeof(λ))`.

# Cost

The log-density takes constant time. Entropy, the CDFs, quantiles, and sampling have
no closed form and sum over the support, which has no upper end, so they stop twenty
standard deviations above the mean. The number of terms grows with `λ`.

Reading that bound off `λ` is a branch on a value, so these five are the one place in
the package a traced or device-side call cannot reach. The closed forms are no better:
the regularized incomplete gamma and a rejection sampler branch on a value too.

The constructor does not check `λ`. A negative rate still gives a finite log-density
at `k = 0`, so validate user input with [`validateparams`](@ref). Use
[`checkparams`](@ref) when only a boolean result is needed.

```julia
checkparams(Poisson(-1.0))               # false
logdensityof(Poisson(-1.0), 0.0)         # finite, and wrong
isnan(logdensityof(Poisson(-1.0), 1.0))  # true
```
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

The support has no upper end, so the sums stop twenty standard deviations above the
mean. What is left out is smaller than the rounding error of a `BigFloat` sum at the
default precision, and far smaller than that of a `Float64` one.

The bound is a plain `Int` read off the value of `λ`, which is what keeps the summing
methods out of traced and device-side code. A rate that is invalid or too large to
count gives a single-term horizon; its density is already non-finite.
"""
@inline function horizon(d::Poisson)
    λ = float(d.λ)
    n = λ + 20 * sqrt(max(λ, zero(λ))) + 60
    return (isfinite(n) & (n > zero(n))) ? ceil(Int, n) : 0
end

@inline function DensityInterface.logdensityof(d::Poisson, x::Number)
    T = masstype(d, x)
    λ, k = convert(T, d.λ), convert(T, x)
    # Skip the zero term so `λ = 0` works.
    a = select(k == zero(T), () -> zero(T), () -> k * logt(λ))
    # Clamp the count because `loggamma` rejects negative integers.
    g = loggamma(max(k, zero(T)) + one(T))
    #=
      `log(k!)` overflows before `k log λ` does and outgrows it in any case, so a count
      that overflows it carries no mass. Returning `-Inf` there keeps the difference
      from becoming `Inf - Inf`.
    =#
    return select(insupport(d, x) & isfinite(g), () -> a - λ - g, () -> convert(T, -Inf))
end

# Inverting the CDF costs no more than one pass over the horizon, and unlike the
# textbook product-of-uniforms draw it does not lose every sample to underflow once
# `exp(-λ)` rounds to zero.
@inline function Base.rand(rng::AbstractRNG, d::Poisson)
    return quantile(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Poisson) = d.λ
Statistics.var(d::Poisson) = d.λ

# No closed form, so sum over the support.
function entropy(d::Poisson)
    T = eltype(d)
    h = zero(T)
    for k in 0:horizon(d)
        logp = logdensityof(d, convert(T, k))
        # An outcome with zero probability contributes zero, not `0 * -Inf = NaN`.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return h
end

# Sum each tail directly so a small tail is not lost by subtracting from one. Clamp
# the result because rounding can make the sum slightly greater than one.
function cdf(d::Poisson, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:horizon(d)
        c += select(k <= x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

# Beyond the horizon this returns zero rather than the true remaining mass.
function ccdf(d::Poisson, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:horizon(d)
        c += select(k > x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

# Count every partial sum below `q` rather than stopping at the first one that is not,
# which keeps the loop length independent of `q`. Summing in the same order as `cdf`
# makes `quantile(d, cdf(d, k))` return `k` unless the CDF has already rounded to one.
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
    # Copy the count before capturing it: a closure over a variable the loop assigns
    # puts that variable on the heap.
    below = i
    # A probability at or above one has no finite answer, so return the last outcome
    # the sum reaches.
    return select(q >= one(T), () -> hi, () -> min(below, hi))
end

function Base.show(io::IO, d::Poisson)
    return print(io, "Poisson(λ=", d.λ, ")")
end
