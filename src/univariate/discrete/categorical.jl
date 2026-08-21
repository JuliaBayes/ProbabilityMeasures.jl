"""
    Categorical(p)

Categorical measure over `1:length(p)`, with probability mass function

```math
P(X = i) = p_i.
```

`p` may be any `AbstractVector{<:Number}`. Its entries must be non-negative and
sum to one. Construction does not validate these conditions; use [`checkparams`](@ref)
for user-supplied probabilities.

Draws and quantiles are category values represented in `eltype(d)`, which is
`float(eltype(p))`.

```julia
d = Categorical([0.2, 0.3, 0.5])
logdensityof(d, 2.0)
rand(d)
```

Negative probabilities produce a non-finite log-density. Values outside the support
have log-density `-Inf`. An unnormalized vector must be detected with `checkparams`.
"""
struct Categorical{V<:AbstractVector{<:Number}} <: DiscreteUnivariateMeasure
    p::V
end

Base.eltype(::Type{Categorical{V}}) where {V} = float(eltype(V))

function checkparams(d::Categorical)
    T = eltype(d)
    total = zero(T)
    # `min` propagates a `NaN` entry without changing numeric types.
    least = zero(T)
    for pᵢ in d.p
        total += pᵢ
        least = min(least, pᵢ)
    end
    # Use the underlying float tolerance for AD and traced element types.
    tol = sqrt(eps(basefloat(T)))
    return (least >= zero(least)) & (abs(total - one(total)) <= tol)
end

support(d::Categorical) = IntegerRange(1, length(d.p))

@inline function masstype(::Categorical{V}, x::Number) where {V}
    return float(promote_type(eltype(V), typeof(x)))
end

"""
    massat(d::Categorical, x)

Return the probability of category `x`, or zero outside the support.

The masked sum also works when `x` is traced and cannot be used as an array index.
"""
@inline function massat(d::Categorical, x::Number)
    T = masstype(d, x)
    m = zero(T)
    for (i, pᵢ) in enumerate(d.p)
        m += select(x == i, () -> convert(T, pᵢ), () -> zero(T))
    end
    return m
end

@inline function DensityInterface.logdensityof(d::Categorical, x::Number)
    return logt(massat(d, x))
end

# Categorical draws do not have a pathwise derivative.
@inline function Base.rand(rng::AbstractRNG, d::Categorical)
    return quantile(d, rand(rng, noisetype(d)))
end

function Statistics.mean(d::Categorical)
    m = zero(eltype(d))
    for (i, pᵢ) in enumerate(d.p)
        m += i * pᵢ
    end
    return m
end

# Two passes rather than `E[X²] - E[X]²`, which cancels badly for a concentrated `p`.
function Statistics.var(d::Categorical)
    m = mean(d)
    v = zero(m)
    for (i, pᵢ) in enumerate(d.p)
        v += pᵢ * (i - m)^2
    end
    return v
end

function entropy(d::Categorical)
    T = eltype(d)
    h = zero(T)
    for pᵢ in d.p
        # A zero probability contributes nothing, where `p log p` would give `NaN`.
        h -= select(pᵢ > zero(pᵢ), () -> pᵢ * logt(pᵢ), () -> zero(T))
    end
    return h
end

# Sum the upper tail directly rather than subtracting the CDF from one.
function cdf(d::Categorical, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for (i, pᵢ) in enumerate(d.p)
        c += select(i <= x, () -> convert(T, pᵢ), () -> zero(T))
    end
    return c
end

function ccdf(d::Categorical, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for (i, pᵢ) in enumerate(d.p)
        c += select(i > x, () -> convert(T, pᵢ), () -> zero(T))
    end
    return c
end

# Count cumulative sums below `q`; traced values cannot drive an early exit.
function Statistics.quantile(d::Categorical, q::Number)
    T = masstype(d, q)
    total = zero(T)
    i = one(T)
    for pᵢ in d.p
        total += convert(T, pᵢ)
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # Clamp an out-of-range `q` to the final category.
    return min(i, convert(T, length(d.p)))
end

function Base.show(io::IO, d::Categorical)
    return print(io, "Categorical(p=", d.p, ")")
end
