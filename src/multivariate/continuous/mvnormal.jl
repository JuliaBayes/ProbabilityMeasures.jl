"""
    MvNormalFactor

The covariance-factor types accepted by [`MvNormal`](@ref).
"""
const MvNormalFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

"""
    MvNormal(μ, L)

Multivariate normal measure with mean `μ` and covariance `L * L'`.

`L` is a lower-triangular covariance factor. It may be a matrix, a
`LinearAlgebra.Diagonal`, or a `LinearAlgebra.UniformScaling`. For a full matrix,
only the lower triangle is used.

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

In these forms, `σ` contains standard deviations. The constructor does not check its
arguments; use [`validateparams`](@ref) for user input.

`logdensityof` returns a non-finite value for invalid parameters or an argument of
the wrong length. Unlike the univariate measures, it allocates a temporary vector.
"""
struct MvNormal{V<:AbstractVector{<:Number},F<:MvNormalFactor} <:
       ContinuousMultivariateMeasure
    μ::V
    L::F
end

const DiagMvNormal = MvNormal{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormal = MvNormal{<:AbstractVector{<:Number},<:UniformScaling}

# Samples are vectors regardless of how the parameters are stored.
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

Return ``\\log \\det L`` in `float(T)`. A non-positive diagonal entry gives a
non-finite result instead of an error.
"""
function logdetfactor(d::MvNormal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for i in 1:length(d.μ)
        acc += logt(convert(R, d.L[i, i]))
    end
    return acc
end

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

# Wrapped numeric comparisons require `&`; shape checks still return ordinary booleans.
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
    # Zero times infinity is `NaN`, which must remain visible in the result.
    acc = L[i, 1] * v[1]
    for j in 2:k
        acc = muladd(L[i, j], v[j], acc)
    end
    return acc
end

"""
    whiten(d::MvNormal, x)

Return ``L^{-1}(x - \\mu)`` using forward substitution.

This builds new arrays instead of modifying one so differentiation tools can follow
the calculation.
"""
function whiten(d::MvNormal, x::AbstractVector{<:Number})
    μ, L = d.μ, d.L
    # The first row sets the result type and keeps later dot products non-empty.
    z = [(x[1] - μ[1]) / L[1, 1]]
    for i in 2:length(μ)
        z = vcat(z, (x[i] - μ[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

whiten(d::DiagMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.diag
whiten(d::IsoMvNormal, x::AbstractVector{<:Number}) = (x .- d.μ) ./ d.L.λ

"""
    unwhiten(d::MvNormal, z)

Convert `z` back to the original coordinates: ``\\mu + L z``.
"""
function unwhiten(d::MvNormal, z::AbstractVector{<:Number})
    return map(i -> d.μ[i] + rowdot(d.L, z, i, i), 1:length(d.μ))
end

unwhiten(d::DiagMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.diag .* z
unwhiten(d::IsoMvNormal, z::AbstractVector{<:Number}) = d.μ .+ d.L.λ .* z

# Result type used when `x` has the wrong length.
@inline function _densitytype(d::MvNormal, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(_densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    # Convert `log2π` explicitly because exact inputs can leave `q` rational.
    R = float(typeof(q))
    n = length(d.μ)
    return -(q + n * convert(R, log2π)) / 2 - logdetfactor(d, R)
end

# Draw plain noise `z`, then return `μ + Lz`.
function Base.rand(rng::AbstractRNG, d::MvNormal)
    return unwhiten(d, randn(rng, noisetype(d), length(d.μ)))
end

Statistics.mean(d::MvNormal) = d.μ

Statistics.cov(d::MvNormal) = (F=LowerTriangular(d.L); F * F')

Statistics.cov(d::DiagMvNormal) = Diagonal(abs2.(d.L.diag))
Statistics.cov(d::IsoMvNormal) = Diagonal(fill(abs2(d.L.λ), length(d.μ)))

# Compute marginal variances without building the full covariance matrix.
Statistics.var(d::MvNormal) = map(i -> sum(j -> abs2(d.L[i, j]), 1:i), 1:length(d.μ))
Statistics.var(d::DiagMvNormal) = abs2.(d.L.diag)
Statistics.var(d::IsoMvNormal) = fill(abs2(d.L.λ), length(d.μ))

Statistics.std(d::MvNormal) = sqrt.(var(d))

function entropy(d::MvNormal)
    logdetL = logdetfactor(d, float(_elparamtype(d)))
    # Keep the constant in the same type as `logdetL`.
    return length(d.μ) * (log2π + one(logdetL)) / 2 + logdetL
end

@inline _elparamtype(d::MvNormal) = promote_type(eltype(d.μ), eltype(d.L))

function Base.show(io::IO, d::MvNormal)
    return print(io, "MvNormal(μ=", d.μ, ", L=", d.L, ")")
end
