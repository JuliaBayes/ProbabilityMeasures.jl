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
struct MvNormal{V<:AbstractVector{<:Number},F<:ScaleFactor} <: MvLocationScale
    μ::V
    L::F
end

const DiagMvNormal = MvNormal{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormal = MvNormal{<:AbstractVector{<:Number},<:UniformScaling}

# Samples are vectors regardless of how the parameters are stored.
function Base.eltype(::Type{MvNormal{V,F}}) where {V,F}
    return Vector{float(promote_type(eltype(V), eltype(F)))}
end

checkparams(d::MvNormal) = checkfactor(d.μ, d.L)

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(densitytype(d, x), NaN)
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

Statistics.cov(d::MvNormal) = scalematrix(d)

Statistics.var(d::MvNormal) = scalediag(d)

function entropy(d::MvNormal)
    logdetL = logdetfactor(d, float(_promoted_paramtype(typeof(d))))
    # Keep the constant in the same type as `logdetL`.
    return length(d.μ) * (log2π + one(logdetL)) / 2 + logdetL
end

function Base.show(io::IO, d::MvNormal)
    return print(io, "MvNormal(μ=", d.μ, ", L=", d.L, ")")
end
