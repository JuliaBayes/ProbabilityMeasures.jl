"""
    Geometric(p)
    Geometric()

The geometric measure on ``\\{0, 1, \\ldots\\}`` with success probability `p`. It
counts failures before the first success, with probability mass function

```math
P(X = k) = p(1-p)^k.
```

`Geometric()` uses `p = 0.5` in `Float64`.

# Arguments

  - `p::Number`: the probability of success, with ``0 < p \\le 1``.

Samples use the floating-point type of `p`. The constructor does not check `p`; use
[`validateparams`](@ref) for user input or [`checkparams`](@ref) for a boolean result.
"""
struct Geometric{P<:Number} <: DiscreteUnivariateMeasure
    p::P
end

Geometric() = Geometric(0.5)

Base.eltype(::Type{Geometric{P}}) where {P} = float(P)

checkparams(d::Geometric) = isfinite(d.p) & (d.p > zero(d.p)) & (d.p <= one(d.p))

support(::Geometric) = NonNegativeIntegers()

@inline function DensityInterface.logdensityof(d::Geometric, x::Number)
    T = masstype(d, x)
    k, p = convert(T, x), convert(T, d.p)
    tail = select(k == zero(T), () -> zero(T), () -> k * log1pt(-p))
    return select(insupport(d, x), () -> logt(p) + tail, () -> convert(T, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::Geometric)
    T = eltype(d)
    u = rand(rng, noisetype(d))
    p = convert(T, d.p)
    return floor(log1pt(-u) / log1pt(-p))
end

Statistics.mean(d::Geometric) = (one(d.p) - d.p) / d.p
Statistics.var(d::Geometric) = (one(d.p) - d.p) / d.p^2

function entropy(d::Geometric)
    T = eltype(d)
    p = convert(T, d.p)
    q = one(T) - p
    tail = select(q > zero(T), () -> q * log1pt(-p) / p, () -> zero(T))
    return -logt(p) - tail
end

function cdf(d::Geometric, x::Number)
    T = masstype(d, x)
    p, k = convert(T, d.p), floor(convert(T, x))
    logtail = (k + one(T)) * log1pt(-p)
    return select(x >= zero(x), () -> -expm1(logtail), () -> zero(T))
end

function ccdf(d::Geometric, x::Number)
    T = masstype(d, x)
    p, k = convert(T, d.p), floor(convert(T, x))
    logtail = (k + one(T)) * log1pt(-p)
    return select(x >= zero(x), () -> exp(logtail), () -> one(T))
end

function logcdf(d::Geometric, x::Number)
    T = masstype(d, x)
    p, k = convert(T, d.p), floor(convert(T, x))
    logtail = (k + one(T)) * log1pt(-p)
    return select(x >= zero(x), () -> log1mexpt(logtail), () -> convert(T, -Inf))
end

function logccdf(d::Geometric, x::Number)
    T = masstype(d, x)
    p, k = convert(T, d.p), floor(convert(T, x))
    logtail = (k + one(T)) * log1pt(-p)
    return select(x >= zero(x), () -> logtail, () -> zero(T))
end

function Statistics.quantile(d::Geometric, q::Number)
    T = masstype(d, q)
    p, probability = convert(T, d.p), convert(T, q)
    k = floor(log1pt(-probability) / log1pt(-p))
    previous_cdf = -expm1(k * log1pt(-p))
    k -= select(probability <= previous_cdf, () -> one(T), () -> zero(T))
    return select(p == one(T), () -> zero(T), () -> max(k, zero(T)))
end

function Base.show(io::IO, d::Geometric)
    return print(io, "Geometric(p=", d.p, ")")
end
