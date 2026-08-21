"""
    Categorical(p)

The categorical measure on ``\\{1, 2, \\ldots, k\\}`` for ``k`` = `length(p)`, with mass
function

```math
P(X = i) = p_i
```

with respect to counting measure.

# Arguments

  - `p::AbstractVector{<:Number}`: the category probabilities, in order.

The `Number` element bound permits numeric wrappers used by AD and tracing systems, and
any `AbstractVector` will do. A plain `Vector` makes the measure non-`isbits`, so a GPU
kernel cannot capture it; an `isbits` container such as a `StaticArrays.SVector` can.

Draws and quantiles are category indices returned in `eltype(d)`, the float type the
probabilities promote to. Keeping them floats is what lets the same code serve AD and
tracing backends, where an index cannot address memory.

Following Distributions.jl, `p` is *required* to be a probability vector rather than
normalized on construction, and [`checkparams`](@ref) is what validates it. Summing to
one is the one part of the contract the density cannot also enforce: no type-generic
tolerance can tell an assembled probability vector from an unnormalized one at every
precision. So an unnormalized `p` gives a finite density that is wrong by a constant, and
user-supplied probabilities have to be checked.

Construction does not validate. A negative probability gives `NaN`, and an `x` carrying
no mass gives `-Inf`.

```julia
checkparams(Categorical([0.5, 0.5, 0.5]))            # false
isnan(logdensityof(Categorical([-1.0, 2.0]), 1.0))   # true
logdensityof(Categorical([0.5, 0.5]), 1.5) == -Inf   # true
```
"""
struct Categorical{V<:AbstractVector{<:Number}} <: DiscreteUnivariateMeasure
    p::V
end

#=
  Julia's generated outer constructor preserves the vector type. Validation is handled
  by `checkparams`.
=#

Base.eltype(::Type{Categorical{V}}) where {V} = float(eltype(V))

function checkparams(d::Categorical)
    T = eltype(d)
    total = zero(T)
    #=
      Track the smallest entry rather than a running `Bool`: the loop stays in one
      numeric type, and `min` carries a `NaN` entry through to the result.
    =#
    least = zero(T)
    for pᵢ in d.p
        total += pᵢ
        least = min(least, pᵢ)
    end
    #=
      `isapprox`'s default tolerance, spelled out because `isapprox` falls back to an
      exact comparison for an AD or traced element type, which no assembled probability
      vector meets. An empty `p` totals zero and fails here.
    =#
    tol = sqrt(eps(basefloat(T)))
    return (least >= zero(least)) & (abs(total - one(total)) <= tol)
end

support(d::Categorical) = IntegerRange(1, length(d.p))

"""
    masstype(d::Categorical, x)

The float type a probability at `x` promotes to: the element type of `d.p` promoted with
`typeof(x)`.
"""
@inline function masstype(::Categorical{V}, x::Number) where {V}
    return float(promote_type(eltype(V), typeof(x)))
end

"""
    massat(d::Categorical, x)

The probability of category `x`, and zero when `x` is not one of `1:length(d.p)`.

A masked sum rather than an indexed load: an index derived from a traced or AD-wrapped
`x` cannot address memory, and every entry other than `x` contributes zero.
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
    #=
      `logt` covers the remaining cases: zero mass, whether from a non-integer `x`, an
      `x` outside `1:k` or an empty `p`, gives `-Inf`, and a negative probability gives
      `NaN`.
    =#
    return logt(massat(d, x))
end

#=
  Inverse-cdf sampling. There is no reparameterization to be had: a category index is a
  piecewise-constant function of `p`, so a draw has no pathwise derivative. Gradients
  with respect to `p` come from the log-density instead.
=#
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

#=
  Masked sums again, for the same reason as `massat`. `ccdf` sums from the far end rather
  than subtracting `cdf` from one, saving a rounding step in the upper tail.

  A sum of non-negative terms is as accurate as its largest term, so `log` of either is
  too, and the generic `logcdf` and `logccdf` fallbacks need no rewriting here, unlike
  the ones `Normal` and `Exponential` replace.
=#
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

#=
  The smallest category whose cumulative mass reaches `q`. Counting the partial sums that
  fall short of `q` replaces the usual early exit, which a traced or vectorized backend
  cannot take.

  `cdf` builds the same partial sums in the same order, differing only by added zeros, so
  `quantile(d, cdf(d, i))` recovers `i` exactly rather than to within a rounding step.

  The argument is `q`, not `p`: `p` is the parameter.
=#
function Statistics.quantile(d::Categorical, q::Number)
    T = masstype(d, q)
    total = zero(T)
    i = one(T)
    for pᵢ in d.p
        total += convert(T, pᵢ)
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    #=
      Total for a `q` outside `[0, 1]`, which can arrive from float noise in a `cdf`
      round-trip, and for an unnormalized `p`: either would otherwise run past the last
      category.
    =#
    return min(i, convert(T, length(d.p)))
end

function Base.show(io::IO, d::Categorical)
    return print(io, "Categorical(p=", d.p, ")")
end
