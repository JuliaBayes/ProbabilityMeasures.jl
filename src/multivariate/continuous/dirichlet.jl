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

Sampling breaks the unit interval into pieces: entry `i` takes the fraction of what is
left that a [`Beta`](@ref) draw with shapes ``\\alpha_i`` and ``\\sum_{j>i} \\alpha_j``
assigns to it. Every piece comes from inverting a continued fraction, so the whole draw
is a smooth function of `α` and carries its derivative.

Sampling inherits [`Beta`](@ref)'s differentiation limits: use ForwardDiff or Enzyme,
not Zygote or Mooncake, which lose the derivative with respect to every shape but the
first. The density itself differentiates under all of them.

`cdf`, `quantile`, `median`, and `entropy` in the univariate sense are not defined here.
Points outside the support, including vectors of the wrong length, have log-density
`-Inf`.
"""
struct Dirichlet{V<:AbstractVector{<:Number}} <: ContinuousMultivariateMeasure
    α::V
end

# Samples are vectors regardless of how the parameters are stored.
Base.eltype(::Type{Dirichlet{V}}) where {V} = Vector{float(eltype(V))}

function checkparams(d::Dirichlet)
    isempty(d.α) && return false
    ok = true
    for αᵢ in d.α
        ok &= isfinite(αᵢ) & (αᵢ > zero(αᵢ))
    end
    return ok
end

support(d::Dirichlet) = RealSimplex(length(d.α))

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
        # `loggamma` throws below zero, so call it at one when `α` is invalid.
        a = pick(valid, convert(T, αᵢ), one(T))
        total += a
        logb += loggamma(a)
    end
    return pick(valid, logb - loggamma(total), convert(T, NaN))
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
    return pick(insupport(d, x) & checkparams(d), logp, convert(T, -Inf))
end

#=
  Break the unit interval into pieces. Entry `i` takes the fraction of the remaining
  stick that a `Beta(αᵢ, Σ_{j>i} αⱼ)` draw assigns to it, which leaves the entries
  Dirichlet distributed and each one differentiable in `α`.
=#
function Base.rand(rng::AbstractRNG, d::Dirichlet)
    T = eltype(eltype(d))
    tail = zero(T)
    for αᵢ in d.α
        tail += convert(T, αᵢ)
    end
    #=
      Grow the vector rather than fill one in place: reverse-mode backends reject array
      mutation, as they do in `MvNormal`'s whitening.
    =#
    x = T[]
    remaining = one(T)
    for i in firstindex(d.α):(lastindex(d.α) - 1)
        αᵢ = convert(T, d.α[i])
        tail -= αᵢ
        piece = remaining * rand(rng, Beta(αᵢ, tail))
        x = vcat(x, piece)
        remaining -= piece
    end
    # The final entry is whatever the earlier ones left, so the draw sums to one.
    return vcat(x, remaining)
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
    a0 = pick(valid, total, one(T))
    h = logmvbeta(d, T) + (a0 - convert(T, length(d.α))) * digamma(a0)
    for αᵢ in d.α
        a = pick(valid, convert(T, αᵢ), one(T))
        h -= (a - one(T)) * digamma(a)
    end
    return pick(valid, h, convert(T, NaN))
end

function Base.show(io::IO, d::Dirichlet)
    return print(io, "Dirichlet(α=", d.α, ")")
end
