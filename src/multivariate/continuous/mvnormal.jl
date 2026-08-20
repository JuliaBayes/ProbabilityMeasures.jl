"""
    MvNormalFactor

The covariance-factor types accepted by [`MvNormal`](@ref).
"""
const MvNormalFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

"""
    MvNormal(μ, L)

Multivariate normal measure with mean `μ` and covariance `L * L'`.

`L` is a lower-triangular covariance factor. It may be a matrix, a
`LinearAlgebra.Diagonal`, or a `LinearAlgebra.UniformScaling`. Only the lower
triangle of a matrix is used.

The density is

```math
p(x) = (2\\pi)^{-n/2} \\left(\\prod_i L_{ii}\\right)^{-1}
       \\exp\\left(-\\frac{1}{2} \\lVert L^{-1}(x - \\mu) \\rVert^2\\right).
```

If you have a covariance matrix `Σ`, factor it before construction:

```julia
using LinearAlgebra
d = MvNormal(μ, Matrix(cholesky(Σ).L))
```

Diagonal and isotropic factors can be written as:

```julia
MvNormal(μ, Diagonal(σ))
MvNormal(μ, σ * I)
```

In these forms, `σ` contains standard deviations. Construction does not validate
the parameters; use [`checkparams`](@ref) when validation is required.

`logdensityof` returns a non-finite value for invalid parameters or an argument of
the wrong length. Unlike the univariate measures, it allocates a temporary vector.
"""
struct MvNormal{V<:AbstractVector{<:Number},F<:MvNormalFactor} <:
       ContinuousMultivariateMeasure
    μ::V
    L::F
end

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`.
=#

#=
  Aliases for dispatch on diagonal and isotropic factors. The vector bound keeps these
  signatures narrower than `MvNormal`.
=#
const DiagMvNormal = MvNormal{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormal = MvNormal{<:AbstractVector{<:Number},<:UniformScaling}

# Draws are vectors regardless of the parameter containers.
function Base.eltype(::Type{MvNormal{V,F}}) where {V,F}
    return Vector{float(promote_type(eltype(V), eltype(F)))}
end

"""
    shapesmatch(d::MvNormal, x) -> Bool

Whether `x`, `μ`, and `L` have compatible dimensions.
"""
@inline function shapesmatch(d::MvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (size(d.L, 1) == n) & (size(d.L, 2) == n)
end

@inline function shapesmatch(d::DiagMvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1) & (length(d.L.diag) == n)
end

@inline function shapesmatch(d::IsoMvNormal, x::AbstractVector)
    n = length(d.μ)
    return (length(x) == n) & (n >= 1)
end

"""
    logdetfactor(d::MvNormal, ::Type{T}) -> float(T)

Return ``\\log \\det L`` in `float(T)`. `logt` makes the result non-finite rather
than throwing when a diagonal entry is non-positive.
"""
function logdetfactor(d::MvNormal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for i in 1:length(d.μ)
        acc += logt(convert(R, d.L[i, i]))
    end
    return acc
end

#=
  Read the stored diagonal directly so reverse-mode AD stays in ordinary array
  arithmetic.
=#
function logdetfactor(d::DiagMvNormal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for σ in d.L.diag
        acc += logt(convert(R, σ))
    end
    return acc
end

function logdetfactor(d::IsoMvNormal, ::Type{T}) where {T}
    return length(d.μ) * logt(convert(float(T), d.L.λ))
end

# Traced comparisons require `&`; shape checks still return ordinary `Bool`s.
function checkparams(d::MvNormal)
    n = length(d.μ)
    (n >= 1) & (size(d.L, 1) == n) & (size(d.L, 2) == n) || return false
    ok = all(isfinite, d.μ)
    for i in 1:n
        Lii = d.L[i, i]
        ok &= isfinite(Lii) & (Lii > zero(Lii))
        for j in 1:(i - 1)
            ok &= isfinite(d.L[i, j])
        end
    end
    return ok
end

function checkparams(d::DiagMvNormal)
    n = length(d.μ)
    (n >= 1) & (length(d.L.diag) == n) || return false
    ok = all(isfinite, d.μ)
    for σ in d.L.diag
        ok &= isfinite(σ) & (σ > zero(σ))
    end
    return ok
end

function checkparams(d::IsoMvNormal)
    λ = d.L.λ
    return (length(d.μ) >= 1) & all(isfinite, d.μ) & isfinite(λ) & (λ > zero(λ))
end

support(d::MvNormal) = RealVectors(length(d.μ))

"""
    rowdot(L, v, i, k)

The inner product of the first `k` entries of row `i` of `L` and `v`.
"""
@inline function rowdot(L::AbstractMatrix, v::AbstractVector, i::Integer, k::Integer)
    # A zero entry of `L` against an infinite `v` gives `NaN`, not an infinity.
    acc = L[i, 1] * v[1]
    for j in 2:k
        acc = muladd(L[i, j], v[j], acc)
    end
    return acc
end

"""
    whiten(d::MvNormal, x)

Return ``L^{-1}(x - \\mu)`` using forward substitution.

The implementation avoids array mutation so reverse-mode AD can differentiate it.
"""
function whiten(d::MvNormal, x::AbstractVector{<:Number})
    μ, L = d.μ, d.L
    #=
      The first row determines the result type and keeps later `rowdot` calls non-empty.
    =#
    z = [(x[1] - μ[1]) / L[1, 1]]
    for i in 2:length(μ)
        z = vcat(z, (x[i] - μ[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

# Diagonal factors whiten elementwise.
whiten(d::DiagMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.diag
whiten(d::IsoMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.λ

"""
    unwhiten(d::MvNormal, z)

The inverse of [`whiten`](@ref): ``\\mu + L z``.
"""
function unwhiten(d::MvNormal, z::AbstractVector{<:Number})
    return map(i -> d.μ[i] + rowdot(d.L, z, i, i), 1:length(d.μ))
end

unwhiten(d::DiagMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.diag .* z
unwhiten(d::IsoMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.λ .* z

# Result type for the mismatched-shape path.
@inline function _densitytype(d::MvNormal, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(_densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    #=
      Convert `log2π` to a floating type explicitly because exact inputs can leave `q`
      rational.
    =#
    R = float(typeof(q))
    n = length(d.μ)
    return -(q + n * convert(R, log2π)) / 2 - logdetfactor(d, R)
end

# Reparameterized sampling: `μ + Lz` with untracked noise `z`.
function Base.rand(rng::AbstractRNG, d::MvNormal)
    return unwhiten(d, randn(rng, noisetype(d), length(d.μ)))
end

Statistics.mean(d::MvNormal) = d.μ

Statistics.cov(d::MvNormal) = (F=LowerTriangular(d.L); F * F')

# Preserve diagonal covariance structure.
Statistics.cov(d::DiagMvNormal) = Diagonal(abs2.(d.L.diag))
Statistics.cov(d::IsoMvNormal) = Diagonal(fill(abs2(d.L.λ), length(d.μ)))

# Marginal variances without forming the covariance matrix.
Statistics.var(d::MvNormal) = map(i -> sum(j -> abs2(d.L[i, j]), 1:i), 1:length(d.μ))
Statistics.var(d::DiagMvNormal) = abs2.(d.L.diag)
Statistics.var(d::IsoMvNormal) = fill(abs2(d.L.λ), length(d.μ))

Statistics.std(d::MvNormal) = sqrt.(var(d))

function entropy(d::MvNormal)
    logdetL = logdetfactor(d, float(_elparamtype(d)))
    # `one(logdetL)`, not `1`: `log2π + 1` would evaluate the `Irrational` at Float64.
    return length(d.μ) * (log2π + one(logdetL)) / 2 + logdetL
end

# Scalar type used for summaries.
@inline _elparamtype(d::MvNormal) = promote_type(eltype(d.μ), eltype(d.L))

function Base.show(io::IO, d::MvNormal)
    return print(io, "MvNormal(μ=", d.μ, ", L=", d.L, ")")
end
