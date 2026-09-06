"""
    Dirichlet(α)

The Dirichlet measure on the probability vectors of length `length(α)`, supported on
[`RealSimplex`](@ref)`(length(α))`, with density

```math
p(x) = \\frac{1}{B(\\alpha)} \\prod_i x_i^{\\alpha_i - 1},
\\qquad
B(\\alpha) = \\frac{\\prod_i \\Gamma(\\alpha_i)}{\\Gamma(\\sum_i \\alpha_i)}.
```

The entries of `α` must be finite and positive. The constructor does not check them;
use [`validateparams`](@ref) for user input.

A draw sums to one, so its entries fill a set of one dimension less than the vector
that holds them. The density above is the one with respect to Lebesgue measure on the
first `length(α) - 1` entries, which is what Distributions.jl reports as well.

Density results promote the types of `α` and the evaluation point. Samples use
`float(eltype(α))`. Density evaluation is linear in `length(α)`.

A draw is a vector of independent gamma draws with shapes `α`, each the quantile of a
uniform noise value, divided by their sum. The division is done in log space, so small
shapes do not underflow, and every shape enters through arithmetic, so automatic
differentiation gives the exact reparameterization gradient.

`cdf`, `quantile`, `median`, and `entropy` in the univariate sense are not defined here.
Points outside the support, including vectors of the wrong length, have log-density
`-Inf`.
"""
struct Dirichlet{V<:AbstractVector{<:Number}} <: ContinuousMultivariateMeasure
    α::V
end

# Samples are vectors regardless of how the parameters are stored.
Base.eltype(::Type{Dirichlet{V}}) where {V} = Vector{float(eltype(V))}

Base.size(d::Dirichlet) = (length(d.α),)

function checkparams(d::Dirichlet)
    isempty(d.α) && return false
    ok = true
    for αᵢ in d.α
        ok &= isfinite(αᵢ) & (αᵢ > zero(αᵢ))
    end
    return ok
end

support(d::Dirichlet) = RealSimplex(length(d.α))

# `loggamma` throws below zero, so evaluate it at one when the parameters are invalid.
@inline safeshape(valid, α::Number) = select(valid, () -> α, () -> one(α))

"""
    logmvbeta(d::Dirichlet, ::Type{T})

``\\log B(\\alpha)``, the normalizing constant of `d`, in type `T`.

Returns `NaN` rather than throwing when an entry of `α` is not positive, where
`loggamma` has a pole.
"""
function logmvbeta(d::Dirichlet, ::Type{T}) where {T}
    valid = checkparams(d)
    total = zero(T)
    logb = zero(T)
    for αᵢ in d.α
        a = safeshape(valid, convert(T, αᵢ))
        total += a
        logb += loggamma(a)
    end
    # Copy the loop-assigned values so the closure does not box them.
    value = logb - loggamma(total)
    return select(valid, () -> value, () -> convert(T, NaN))
end

function DensityInterface.logdensityof(d::Dirichlet, x::AbstractVector{<:Number})
    T = float(promote_type(eltype(d.α), eltype(x)))
    length(x) == length(d.α) || return convert(T, -Inf)
    logp = -logmvbeta(d, T)
    # Pair the entries rather than index them, so an argument whose axes differ from
    # `α`'s still evaluates.
    for (αᵢ, xᵢ) in zip(d.α, x)
        logp += xlogyt(convert(T, αᵢ) - one(T), convert(T, xᵢ))
    end
    value = logp
    return select(insupport(d, x) & checkparams(d), () -> value, () -> convert(T, -Inf))
end

#=
  Normalized gamma draws are Dirichlet distributed. Working with the log quantiles
  keeps a small shape from underflowing, and `map` rather than a filled vector keeps
  reverse-mode backends, which reject array mutation, on the same path.
=#
function Base.rand(rng::AbstractRNG, d::Dirichlet)
    T = eltype(eltype(d))
    F = basefloat(T)
    valid = checkparams(d)
    logs = map(d.α) do αᵢ
        a = convert(T, αᵢ)
        return select(valid, () -> gammalogquantile(a, rand(rng, F)), () -> convert(T, NaN))
    end
    shift = maximum(logs)
    w = exp.(logs .- shift)
    return w ./ sum(w)
end

Statistics.mean(d::Dirichlet) = d.α ./ sum(d.α)

function Statistics.var(d::Dirichlet)
    total = sum(d.α)
    return d.α .* (total .- d.α) ./ (total^2 * (total + one(total)))
end

Statistics.std(d::Dirichlet) = sqrt.(var(d))

function Statistics.cov(d::Dirichlet)
    total = sum(d.α)
    scale = total^2 * (total + one(total))
    return [
        d.α[i] * ((i == j ? total : zero(total)) - d.α[j]) / scale for
        i in eachindex(d.α), j in eachindex(d.α)
    ]
end

function entropy(d::Dirichlet)
    T = eltype(eltype(d))
    total = zero(T)
    for αᵢ in d.α
        total += convert(T, αᵢ)
    end
    valid = checkparams(d)
    a0 = safeshape(valid, total)
    h = logmvbeta(d, T) + (a0 - convert(T, length(d.α))) * digamma(a0)
    for αᵢ in d.α
        a = safeshape(valid, convert(T, αᵢ))
        h -= (a - one(T)) * digamma(a)
    end
    value = h
    return select(valid, () -> value, () -> convert(T, NaN))
end

function Base.show(io::IO, d::Dirichlet)
    return print(io, "Dirichlet(α=", d.α, ")")
end
