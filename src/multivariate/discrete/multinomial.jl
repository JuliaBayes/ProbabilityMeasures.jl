"""
    Multinomial(n, p)

The counts in `n` independent categorical trials with probabilities `p`, supported on
[`IntegerSimplex`](@ref)`(n, length(p))`, with probability mass function

```math
P(X = x) = \\frac{n!}{\\prod_i x_i!} \\prod_i p_i^{x_i}.
```

# Arguments

  - `n::Integer`: the number of trials.
  - `p::AbstractVector`: the category probabilities, non-negative and summing to one.

`n` stays an integer because it sets the support and the sampling loop length. The
result type follows `p` and the value being evaluated. Samples use `float(eltype(p))`.

# Cost

The log-density takes time linear in `length(p)`. Sampling loops over `n` trials, each
a categorical draw over `length(p)` outcomes. Both loops work with automatic
differentiation and on GPUs. There is no `cdf`, `quantile`, `median` or `entropy`,
none of which has a closed form in more than one dimension.

An argument of the wrong length gives `-Inf`, where [`MvNormal`](@ref) gives `NaN` for
a shape mismatch. A count vector of the wrong length is simply outside the support.

The constructor does not check its arguments. An invalid `p` can still give a finite
log-density wherever the offending entries carry a zero count, so validate user input
with [`validateparams`](@ref). Use [`checkparams`](@ref) when only a boolean result is
needed.

```julia
d = Multinomial(5, [NaN, 0.4, 0.6])
checkparams(d)                          # false
isnan(logdensityof(d, [1.0, 1.0, 3.0])) # true
logdensityof(d, [0.0, 2.0, 3.0])        # finite, and wrong
```
"""
struct Multinomial{N<:Integer,V<:AbstractVector{<:Number}} <: DiscreteMultivariateMeasure
    n::N
    p::V
end

function Base.eltype(::Type{Multinomial{N,V}}) where {N,V}
    return Vector{float(eltype(V))}
end

checkparams(d::Multinomial) = (d.n >= zero(d.n)) & checkparams(Categorical(d.p))

support(d::Multinomial) = IntegerSimplex(d.n, length(d.p))

function DensityInterface.logdensityof(d::Multinomial, x::AbstractVector{<:Number})
    T = float(promote_type(typeof(d.n), eltype(d.p), eltype(x)))
    length(x) == length(d.p) || return convert(T, -Inf)
    n = convert(T, d.n)
    logp = loggamma(max(n, zero(T)) + one(T))
    for i in eachindex(d.p)
        xᵢ, pᵢ = convert(T, x[i]), convert(T, d.p[i])
        logp -= loggamma(max(xᵢ, zero(T)) + one(T))
        logp += select(xᵢ == zero(T), () -> zero(T), () -> xᵢ * logt(pᵢ))
    end
    # Binding the accumulator keeps the closure below from boxing it, so this is 0-alloc.
    result = logp
    return select(insupport(d, x), () -> result, () -> convert(T, -Inf))
end

function Base.rand(rng::AbstractRNG, d::Multinomial)
    T = eltype(eltype(d))
    counts = zeros(T, length(d.p))
    categorical = Categorical(d.p)
    for _ in 1:(d.n)
        category = rand(rng, categorical)
        for i in eachindex(counts)
            counts[i] += select(category == i, () -> one(T), () -> zero(T))
        end
    end
    return counts
end

Statistics.mean(d::Multinomial) = d.n .* d.p

function Statistics.cov(d::Multinomial)
    return [
        d.n * d.p[i] * (i == j ? one(d.p[i]) - d.p[j] : -d.p[j]) for
        i in eachindex(d.p), j in eachindex(d.p)
    ]
end

Statistics.var(d::Multinomial) = d.n .* d.p .* (one.(d.p) .- d.p)
Statistics.std(d::Multinomial) = sqrt.(var(d))

function Base.show(io::IO, d::Multinomial)
    return print(io, "Multinomial(n=", d.n, ", p=", d.p, ")")
end
