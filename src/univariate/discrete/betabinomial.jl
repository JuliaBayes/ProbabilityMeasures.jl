"""
    BetaBinomial(n, α, β)

The beta-binomial measure on ``\\{0, 1, \\ldots, n\\}``: the number of successes in `n`
trials that share one success probability drawn from a beta measure with shapes `α` and
`β`. Its probability mass function is

```math
P(X = k) = \\binom{n}{k} \\frac{B(k + \\alpha, n - k + \\beta)}{B(\\alpha, \\beta)},
```

where ``B`` is the beta function.

The shared probability correlates the trials, so the variance is larger than a
binomial's with the same mean, by a factor of
``(\\alpha + \\beta + n)/(\\alpha + \\beta + 1)``. `BetaBinomial(n, 1, 1)` spreads the
mass evenly over the support, and large ``\\alpha + \\beta`` approaches
`Binomial(n, α / (α + β))`.

# Arguments

  - `n::Integer`: the number of trials.
  - `α::Number`: the first shape, positive.
  - `β::Number`: the second shape, positive.

`n` stays an integer because it sets the support and loop lengths. The result type
follows the shapes and the value being evaluated. Samples use
`float(promote_type(typeof(α), typeof(β)))`.

# Cost

The log-density takes constant time. Sampling, entropy, CDFs, and quantiles evaluate
the log-density at each of the `n + 1` outcomes. These loops work with automatic
differentiation and on GPUs.

The log-density is a difference of `loggamma` terms of order ``n \\log n``, so its
relative accuracy falls off roughly in proportion to `n`: about `3e-13` at `n = 1000`
and `2e-9` at `n = 10^7` in `Float64`.

For large `n`, rounding can make the CDF reach one before the last outcome.
`quantile(d, cdf(d, k))` may then fail to recover those final outcomes, though a
probability of one always returns `n`.

The constructor does not check its arguments. A shape that is not positive and finite
gives `NaN` across the whole support, so validate user input with
[`validateparams`](@ref). Use [`checkparams`](@ref) when only a boolean result is
needed.

```julia
checkparams(BetaBinomial(3, -1.0, 2.0))               # false
isnan(logdensityof(BetaBinomial(3, -1.0, 2.0), 1.0))  # true
```
"""
struct BetaBinomial{N<:Integer,A<:Number,B<:Number} <: DiscreteUnivariateMeasure
    n::N
    α::A
    β::B
end

Base.eltype(::Type{BetaBinomial{N,A,B}}) where {N,A,B} = float(promote_type(A, B))

function checkparams(d::BetaBinomial)
    valid = isfinite(d.α) & (d.α > zero(d.α)) & isfinite(d.β) & (d.β > zero(d.β))
    return (d.n >= zero(d.n)) & valid
end

support(d::BetaBinomial) = IntegerRange(0, d.n)

"""
    betabinomial_logmass(n, k, α, β)

The beta-binomial log-density at `k`, with every value already converted to floating
point and `k` known to lie in ``\\{0, \\ldots, n\\}``.

Outside that range one beta argument can be non-positive, which [`logbetat`](@ref)
reports as `NaN` rather than a `DomainError`.

The counts and the shapes are typed separately: the numbers a reverse-mode tape
produces need not share a type with the ones it was given.
"""
@inline function betabinomial_logmass(n::Number, k::Number, α::Number, β::Number)
    return logbinom(n, k) + logbetat(k + α, n - k + β) - logbetat(α, β)
end

@inline function DensityInterface.logdensityof(d::BetaBinomial, x::Number)
    T = masstype(d, x)
    n, k = convert(T, d.n), convert(T, x)
    α, β = convert(T, d.α), convert(T, d.β)
    value = betabinomial_logmass(n, k, α, β)
    return select(insupport(d, x), () -> value, () -> convert(T, -Inf))
end

Statistics.mean(d::BetaBinomial) = d.n * d.α / (d.α + d.β)

function Statistics.var(d::BetaBinomial)
    s = d.α + d.β
    return d.n * d.α * d.β * (s + d.n) / (s^2 * (s + one(s)))
end

function Base.show(io::IO, d::BetaBinomial)
    return print(io, "BetaBinomial(n=", d.n, ", α=", d.α, ", β=", d.β, ")")
end

"""
    BetaBinomialLogit(n, η, ϕ)

The beta-binomial measure on ``\\{0, 1, \\ldots, n\\}`` in the mean-precision
parameterization: `η` is the logit of the mean success probability and `ϕ` is the
precision. It is [`BetaBinomial`](@ref)`(n, α, β)` with

```math
\\alpha = \\phi\\, \\sigma(\\eta), \\qquad \\beta = \\phi\\, \\sigma(-\\eta),
```

where ``\\sigma`` is the logistic function. The shapes sum to `ϕ`, so the mean is
``n\\,\\sigma(\\eta)`` and the variance is
``n\\,\\sigma(\\eta)\\,\\sigma(-\\eta)\\,(n + \\phi)/(1 + \\phi)``. Large `ϕ` approaches
`Binomial(n, σ(η))`, and `ϕ = 2` with `η = 0` spreads the mass evenly.

This is the parameterization a regression uses: `η` is the linear predictor, unbounded
and free to vary, while `ϕ` carries the overdispersion on its own.

# Arguments

  - `n::Integer`: the number of trials.
  - `η::Number`: the logit of the mean success probability.
  - `ϕ::Number`: the precision, positive.

`n` stays an integer because it sets the support and loop lengths. The result type
follows `η`, `ϕ`, and the value being evaluated. Samples use
`float(promote_type(typeof(η), typeof(ϕ)))`.

# Cost

The same as [`BetaBinomial`](@ref): a constant-time log-density, and a loop over the
`n + 1` outcomes for sampling, entropy, CDFs, and quantiles.

The constructor does not check its arguments. [`checkparams`](@ref) tests the implied
shapes rather than `η` and `ϕ` directly, so it also rejects an `η` past `±709` in
`Float64` or `±88` in `Float32`, where `exp(η)` overflows and one shape becomes zero
whatever `ϕ` is. The log-density is `NaN` across the whole support there.

```julia
checkparams(BetaBinomialLogit(3, 0.0, -1.0))                # false
checkparams(BetaBinomialLogit(3, 800.0, 1.0))               # false, `β` is zero
isnan(logdensityof(BetaBinomialLogit(3, 800.0, 1.0), 1.0))  # true
```
"""
struct BetaBinomialLogit{N<:Integer,H<:Number,P<:Number} <: DiscreteUnivariateMeasure
    n::N
    η::H
    ϕ::P
end

Base.eltype(::Type{BetaBinomialLogit{N,H,P}}) where {N,H,P} = float(promote_type(H, P))

"""
    betashapes(η, ϕ) -> (α, β)

The beta shapes `ϕ * logistic(η)` and `ϕ * logistic(-η)`, which sum to `ϕ`.
"""
@inline betashapes(η::Number, ϕ::Number) = (ϕ * logistic(η), ϕ * logistic(-η))

# Promote first, so that the check below uses the range of the wider parameter.
betashapes(d::BetaBinomialLogit) = betashapes(promote(d.η, d.ϕ)...)

# An extreme `η` sends one shape to zero, which no density can report, so the check
# runs on the shapes themselves rather than on `η` and `ϕ`.
function checkparams(d::BetaBinomialLogit)
    α, β = betashapes(d)
    valid = isfinite(α) & (α > zero(α)) & isfinite(β) & (β > zero(β))
    return (d.n >= zero(d.n)) & valid
end

support(d::BetaBinomialLogit) = IntegerRange(0, d.n)

@inline function DensityInterface.logdensityof(d::BetaBinomialLogit, x::Number)
    T = masstype(d, x)
    n, k = convert(T, d.n), convert(T, x)
    # Form the shapes in `T` so that exact parameters keep the argument's precision.
    α, β = betashapes(convert(T, d.η), convert(T, d.ϕ))
    value = betabinomial_logmass(n, k, α, β)
    return select(insupport(d, x), () -> value, () -> convert(T, -Inf))
end

Statistics.mean(d::BetaBinomialLogit) = d.n * logistic(d.η)

function Statistics.var(d::BetaBinomialLogit)
    p, q = logistic(d.η), logistic(-d.η)
    return d.n * p * q * (d.n + d.ϕ) / (one(d.ϕ) + d.ϕ)
end

function Base.show(io::IO, d::BetaBinomialLogit)
    return print(io, "BetaBinomialLogit(n=", d.n, ", η=", d.η, ", ϕ=", d.ϕ, ")")
end

"""
    BetaBinomialForm

Either beta-binomial parameterization. Sampling, entropy and the distribution functions
sum over the same support and depend on the parameters only through the log-density.
"""
const BetaBinomialForm = Union{BetaBinomial,BetaBinomialLogit}

# The trials share one success probability, so they cannot be drawn one at a time.
# Inverting the CDF keeps a draw to a single fixed-length loop over the support.
@inline function Base.rand(rng::AbstractRNG, d::BetaBinomialForm)
    return quantile(d, rand(rng, noisetype(d)))
end

# No closed form, so sum over the support.
function entropy(d::BetaBinomialForm)
    T = eltype(d)
    h = zero(T)
    for k in 0:(d.n)
        logp = logdensityof(d, convert(T, k))
        # An outcome with zero probability contributes zero, not `0 * -Inf = NaN`.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return h
end

# Sum each tail directly so a small tail is not lost by subtracting from one. Clamp
# the result because rounding can make the sum slightly greater than one.
function cdf(d::BetaBinomialForm, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:(d.n)
        c += select(k <= x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

function ccdf(d::BetaBinomialForm, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:(d.n)
        c += select(k > x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

# Some tools cannot stop a loop based on `q`, so count every partial sum below it.
# Using the same order as `cdf` makes `quantile(d, cdf(d, k))` return `k` unless the
# CDF has already rounded to one.
function Statistics.quantile(d::BetaBinomialForm, q::Number)
    T = masstype(d, q)
    n = convert(T, d.n)
    total = zero(T)
    i = zero(T)
    for k in 0:(d.n)
        total += exp(logdensityof(d, convert(T, k)))
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # Copy the loop-assigned value to avoid boxing the closure capture.
    below = i
    # A probability at or above one always returns the last outcome.
    return select(q >= one(T), () -> n, () -> min(below, n))
end
