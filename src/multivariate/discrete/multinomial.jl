"""
    Multinomial(n, p)

The counts in `n` independent categorical trials with probabilities `p`, supported on
[`IntegerSimplex`](@ref)`(n, length(p))`, with probability mass function

```math
P(X = x) = \\frac{n!}{\\prod_i x_i!} \\prod_i p_i^{x_i}.
```

`n` must be non-negative, and the entries of `p` must be non-negative and sum to one.
The constructor does not check these conditions; use [`validateparams`](@ref) for user
input.

Density results follow Julia's promotion rules for `n`, `p`, and the evaluation point.
Samples use `float(eltype(p))`. Density evaluation is linear in `length(p)`; sampling
performs `n` categorical draws. Both operations support automatic differentiation and
GPUs.

`cdf`, `quantile`, `median`, and `entropy` are not implemented. Points outside the
support, including vectors of the wrong length, have log-density `-Inf`.
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
    # Avoid boxing the accumulator in the closure below.
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
