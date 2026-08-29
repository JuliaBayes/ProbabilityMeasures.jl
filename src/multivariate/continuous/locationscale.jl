"""
    MvLocationScale

Supertype for multivariate measures written as ``\\mu + L z``, where `z` is a draw from
a fixed spherical measure.

Subtypes store a location vector `μ` and a lower-triangular scale factor `L`, and share
the whitening, log-determinant, shape and scale-matrix code below. `L` may be a matrix,
a `LinearAlgebra.Diagonal`, or a `LinearAlgebra.UniformScaling`; the three forms
dispatch on the factor rather than on the measure.
"""
abstract type MvLocationScale <: ContinuousMultivariateMeasure end

"""
    ScaleFactor

The scale-factor types accepted by an [`MvLocationScale`](@ref).
"""
const ScaleFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

support(d::MvLocationScale) = RealVectors(length(d.μ))

"""
    shapesmatch(d::MvLocationScale, x) -> Bool

Whether `x`, `μ`, and `L` have compatible dimensions.
"""
@inline shapesmatch(d::MvLocationScale, x::AbstractVector) = shapesmatch(d.μ, d.L, x)

@inline function shapesmatch(μ::AbstractVector, L::AbstractMatrix, x::AbstractVector)
    n = length(μ)
    return (length(x) == n) & (n >= 1) & (size(L, 1) == n) & (size(L, 2) == n)
end

@inline function shapesmatch(μ::AbstractVector, L::Diagonal, x::AbstractVector)
    n = length(μ)
    return (length(x) == n) & (n >= 1) & (length(L.diag) == n)
end

@inline function shapesmatch(μ::AbstractVector, ::UniformScaling, x::AbstractVector)
    n = length(μ)
    return (length(x) == n) & (n >= 1)
end

"""
    logdetfactor(d::MvLocationScale, ::Type{T}) -> float(T)

Return ``\\log \\det L`` in `float(T)`. A non-positive diagonal entry gives a
non-finite result instead of an error.
"""
function logdetfactor(d::MvLocationScale, ::Type{T}) where {T}
    return logdetfactor(length(d.μ), d.L, T)
end

function logdetfactor(n::Int, L::AbstractMatrix, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for i in 1:n
        acc += logt(convert(R, L[i, i]))
    end
    return acc
end

function logdetfactor(::Int, L::Diagonal, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for σ in L.diag
        acc += logt(convert(R, σ))
    end
    return acc
end

function logdetfactor(n::Int, L::UniformScaling, ::Type{T}) where {T}
    return n * logt(convert(float(T), L.λ))
end

"""
    checkfactor(μ, L) -> Bool

Whether `μ` and `L` are a usable location and scale factor: matching dimensions, finite
entries, and a positive diagonal.
"""
function checkfactor(μ::AbstractVector, L::AbstractMatrix)
    n = length(μ)
    # Wrapped numeric comparisons require `&`; shape checks still return ordinary
    # booleans.
    (n >= 1) & (size(L, 1) == n) & (size(L, 2) == n) || return false
    ok = all(isfinite, μ)
    for i in 1:n
        Lii = L[i, i]
        ok &= isfinite(Lii) & (Lii > zero(Lii))
        for j in 1:(i - 1)
            ok &= isfinite(L[i, j])
        end
    end
    return ok
end

function checkfactor(μ::AbstractVector, L::Diagonal)
    n = length(μ)
    (n >= 1) & (length(L.diag) == n) || return false
    ok = all(isfinite, μ)
    for σ in L.diag
        ok &= isfinite(σ) & (σ > zero(σ))
    end
    return ok
end

function checkfactor(μ::AbstractVector, L::UniformScaling)
    λ = L.λ
    return (length(μ) >= 1) & all(isfinite, μ) & isfinite(λ) & (λ > zero(λ))
end

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
    whiten(d::MvLocationScale, x)

Return ``L^{-1}(x - \\mu)`` using forward substitution.

This builds new arrays instead of modifying one so differentiation tools can follow the
calculation.
"""
whiten(d::MvLocationScale, x::AbstractVector{<:Number}) = whiten(d.μ, d.L, x)

function whiten(μ::AbstractVector, L::AbstractMatrix, x::AbstractVector{<:Number})
    # The first row sets the result type and keeps later dot products non-empty.
    z = [(x[1] - μ[1]) / L[1, 1]]
    for i in 2:length(μ)
        z = vcat(z, (x[i] - μ[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

whiten(μ::AbstractVector, L::Diagonal, x::AbstractVector{<:Number}) = (x .- μ) ./ L.diag
whiten(μ::AbstractVector, L::UniformScaling, x::AbstractVector{<:Number}) = (x .- μ) ./ L.λ

"""
    unwhiten(d::MvLocationScale, z)

Convert `z` back to the original coordinates: ``\\mu + L z``.
"""
unwhiten(d::MvLocationScale, z::AbstractVector{<:Number}) = unwhiten(d.μ, d.L, z)

function unwhiten(μ::AbstractVector, L::AbstractMatrix, z::AbstractVector{<:Number})
    return map(i -> μ[i] + rowdot(L, z, i, i), 1:length(μ))
end

unwhiten(μ::AbstractVector, L::Diagonal, z::AbstractVector{<:Number}) = μ .+ L.diag .* z
unwhiten(μ::AbstractVector, L::UniformScaling, z::AbstractVector{<:Number}) = μ .+ L.λ .* z

"""
    scalematrix(d::MvLocationScale)

The matrix ``L L^\\top``.
"""
scalematrix(d::MvLocationScale) = scalematrix(length(d.μ), d.L)

scalematrix(::Int, L::AbstractMatrix) = (F=LowerTriangular(L); F * F')
scalematrix(::Int, L::Diagonal) = Diagonal(abs2.(L.diag))
scalematrix(n::Int, L::UniformScaling) = Diagonal(fill(abs2(L.λ), n))

"""
    scalediag(d::MvLocationScale)

The diagonal of ``L L^\\top``, without building the full matrix.
"""
scalediag(d::MvLocationScale) = scalediag(length(d.μ), d.L)

scalediag(n::Int, L::AbstractMatrix) = map(i -> sum(j -> abs2(L[i, j]), 1:i), 1:n)
scalediag(::Int, L::Diagonal) = abs2.(L.diag)
scalediag(n::Int, L::UniformScaling) = fill(abs2(L.λ), n)

Statistics.std(d::MvLocationScale) = sqrt.(var(d))

# Result type used when `x` has the wrong length.
@inline function densitytype(d::MvLocationScale, x::AbstractVector)
    return float(promote_type(_promoted_paramtype(typeof(d)), eltype(x)))
end
