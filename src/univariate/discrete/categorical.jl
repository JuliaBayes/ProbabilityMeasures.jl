"""
    Categorical(p)

A categorical measure over `1:length(p)` with probability mass function

```math
P(X = i) = p_i.
```

The entries of `p` must be non-negative and sum to one. The constructor does not
check them, so use [`validateparams`](@ref) for user input or [`checkparams`](@ref)
when only a boolean result is needed.

Samples and quantiles use the floating-point type of `p` rather than an integer type.

```julia
d = Categorical([0.2, 0.3, 0.5])
logdensityof(d, 2.0)
rand(d)
```

Values outside the support have log-density `-Inf`. Invalid probabilities may still
give finite results for some categories, which is why explicit validation matters.
"""
struct Categorical{V<:AbstractVector{<:Number}} <: DiscreteUnivariateMeasure
    p::V
end

Base.eltype(::Type{Categorical{V}}) where {V} = float(eltype(V))

function checkparams(d::Categorical)
    T = eltype(d)
    total = zero(T)
    # `min` keeps a `NaN` visible without changing numeric types.
    least = zero(T)
    for pᵢ in d.p
        total += pᵢ
        least = min(least, pᵢ)
    end
    # Use the plain floating-point precision for wrapped element types.
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

The loop also works when `x` is wrapped by a tool and cannot index an array.
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

# A sample changes in steps as the probabilities change, so its derivative is zero
# almost everywhere.
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

# Center the values first; `E[X²] - E[X]²` loses precision when one category dominates.
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
        # A zero probability contributes zero, not `0 * log(0) = NaN`.
        h -= select(pᵢ > zero(pᵢ), () -> pᵢ * logt(pᵢ), () -> zero(T))
    end
    return h
end

# Sum each tail directly so a small upper tail is not lost by subtracting from one.
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

# Some tools cannot stop a loop based on `q`, so count every partial sum below it.
function Statistics.quantile(d::Categorical, q::Number)
    T = masstype(d, q)
    total = zero(T)
    i = one(T)
    for pᵢ in d.p
        total += convert(T, pᵢ)
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # A probability above one still returns the final category.
    return min(i, convert(T, length(d.p)))
end

function Base.show(io::IO, d::Categorical)
    return print(io, "Categorical(p=", d.p, ")")
end
