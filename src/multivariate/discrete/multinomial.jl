"""
    Multinomial(n, p)

A multinomial measure over `1:length(p)`, with probability mass function

```math
P(X = x) = \\frac{n!}{x_1! \\cdots x_k!} \\prod_{i=1}^{k} p_i^{x_i}.
```

The entries of `p` must be non-negative and sum to one.

# Arguments

  - `n::Integer`: the number of draws.
  - `p::AbstractVector`: the category probabilities, non-negative and summing to one.

`n` stays an integer because it sets the support and the sampling loop length. The
result type follows `p` and the value being evaluated. Draws are count vectors in the
floating-point type of `p`, as for [`Categorical`](@ref) and [`Binomial`](@ref).

# Cost

The log-density is linear in the number of categories. A draw counts `n` categorical
samples, so it costs `n` times that. Both are fixed-length loops, so they work with
automatic differentiation and on GPUs.

There is no `cdf`, `quantile` or `entropy`: none has a closed form, and summing over
the support means enumerating every count vector that sums to `n`.

The constructor does not check its arguments. Probabilities that do not sum to one
still give a finite log-density, so validate user input with [`validateparams`](@ref).
Use [`checkparams`](@ref) when only a boolean result is needed.

```julia
d = Multinomial(4, [0.2, 0.3, 0.5])
logdensityof(d, [2.0, 1.0, 1.0])
rand(d)
```
"""
struct Multinomial{N<:Integer,V<:AbstractVector{<:Number}} <: DiscreteMultivariateMeasure
    n::N
    p::V
end

# Samples are count vectors regardless of how the probabilities are stored.
Base.eltype(::Type{Multinomial{N,V}}) where {N,V} = Vector{float(eltype(V))}

# The probabilities carry the same requirements as those of a `Categorical`.
function checkparams(d::Multinomial)
    return (d.n >= zero(d.n)) & checkparams(Categorical(d.p))
end

support(d::Multinomial) = CountVectors(d.n, length(d.p))

"""
    logmassat(d::Multinomial, x)

Return the log probability of the count vector `x` without checking the support.

The counts are clamped because `loggamma` rejects negative non-integers. Clamping only
affects values outside the support, where [`logdensityof`](@ref) returns `-Inf`.

The sum stays out of `logdensityof` because a loop-updated variable captured by a
closure there would be boxed, which costs inference.
"""
@inline function logmassat(d::Multinomial, x::AbstractVector{<:Number})
    T = masstype(d, x)
    acc = loggamma(max(convert(T, d.n), zero(T)) + one(T))
    for (pᵢ, xᵢ) in zip(d.p, x)
        xt = convert(T, xᵢ)
        acc -= loggamma(max(xt, zero(T)) + one(T))
        # Skip zero counts so a category of probability zero works.
        acc += select(xt == zero(T), () -> zero(T), () -> xt * logt(convert(T, pᵢ)))
    end
    return acc
end

function DensityInterface.logdensityof(d::Multinomial, x::AbstractVector{<:Number})
    T = masstype(d, x)
    length(x) == length(d.p) || return convert(T, -Inf)
    m = logmassat(d, x)
    return select(insupport(d, x), () -> m, () -> convert(T, -Inf))
end

#=
  Count `n` categorical draws. This fixed-length loop also works with tracing tools
  and on GPUs, unlike the conditional-binomial construction, whose trial counts depend
  on the draws already taken.
=#
function Base.rand(rng::AbstractRNG, d::Multinomial)
    T = float(eltype(d.p))
    counts = zeros(T, length(d.p))
    categories = Categorical(d.p)
    for _ in 1:(d.n)
        j = quantile(categories, rand(rng, noisetype(d)))
        for i in eachindex(counts)
            counts[i] += select(j == i, () -> one(T), () -> zero(T))
        end
    end
    return counts
end

Statistics.mean(d::Multinomial) = d.n .* d.p

Statistics.var(d::Multinomial) = d.n .* d.p .* (one(eltype(d.p)) .- d.p)

# Counts sum to `n`, so a category gaining mass takes it from the others.
function Statistics.cov(d::Multinomial)
    n, p = d.n, d.p
    return [n * p[i] * ((i == j) - p[j]) for i in eachindex(p), j in eachindex(p)]
end

Statistics.std(d::Multinomial) = sqrt.(var(d))

function Base.show(io::IO, d::Multinomial)
    return print(io, "Multinomial(n=", d.n, ", p=", d.p, ")")
end
