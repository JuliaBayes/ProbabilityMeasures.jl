"""
    MvNormalFactor

The factor types [`MvNormal`](@ref) accepts: any matrix, or a `UniformScaling`, which is
not one because it carries no size of its own.
"""
const MvNormalFactor = Union{AbstractMatrix{<:Number},UniformScaling{<:Number}}

"""
    MvNormal(μ, L)

The multivariate normal (Gaussian) measure on ``\\mathbb{R}^n`` with mean `μ` and
covariance ``\\Sigma = L L^{\\top}``, with density

```math
p(x) = (2\\pi)^{-n/2} \\left(\\textstyle\\prod_i L_{ii}\\right)^{-1}
       \\exp\\!\\left(-\\tfrac{1}{2} \\lVert L^{-1}(x - \\mu) \\rVert^2\\right)
```

with respect to Lebesgue measure on ``\\mathbb{R}^n``. Draws are `Vector`s.

# Arguments

  - `μ::AbstractVector{<:Number}`: the mean.
  - `L`: the lower-triangular Cholesky factor of the covariance, as a matrix, a
    `LinearAlgebra.Diagonal`, or a `LinearAlgebra.UniformScaling`. Only the lower
    triangle of a matrix is read, as with `LinearAlgebra.LowerTriangular`.

The `Number` element bound permits numeric wrappers used by AD and tracing systems.

# Parameterization

The factor, not the covariance. A draw is ``\\mu + L z`` for standard normal ``z``, so
the reparameterized sample and its pathwise gradient are plain arithmetic in the
parameters, and the log-density costs one triangular solve rather than a factorization
of ``\\Sigma`` per evaluation. A caller holding a covariance factors it once:

```julia
using LinearAlgebra
MvNormal(μ, Matrix(cholesky(Σ).L))
```

There is no `cdf`, `quantile` or `median`: none of them has a closed form in more than
one dimension.

# Structured factors

A `Diagonal` or `UniformScaling` factor takes a shorter path. Whitening a general factor
is a forward substitution, ``O(n^2)`` and sequential; a diagonal one is a single
elementwise division. It is the same measure, computed more cheaply, and the two paths
agree to floating-point rounding rather than bit for bit: an isotropic
``\\log \\det L`` is ``n \\log \\lambda`` where the general path adds ``\\log \\lambda``
``n`` times, and for larger ``n`` the two can land a last bit apart.

```julia
using LinearAlgebra

MvNormal(μ, L)                # general, one triangular solve per density evaluation
MvNormal(μ, Diagonal(σ))      # independent coordinates, σ their standard deviations
MvNormal(μ, σ * I)            # isotropic, σ the common standard deviation
```

!!! warning
    The second argument is always the *factor*, so a structured one carries standard
    deviations. Distributions.jl spells its structured cases with the *covariance*, as
    `MvNormal(μ, σ² * I)`. Take a square root when porting.

A `UniformScaling` carries no dimension, so the isotropic form takes ``n`` from `μ`.
Both structured factors are `isbits`, so an isotropic measure over a statically sized
mean is capturable by a device kernel where a matrix-backed one is not.

Unlike the univariate measures, `logdensityof` allocates. Whitening ``x - \\mu`` needs
a temporary, written so that reverse-mode AD can differentiate it, and a draw has to
allocate the vector it returns.

# Examples

Construction does not validate. Invalid parameters produce a non-finite density; use
[`checkparams`](@ref) to validate explicitly.

```julia
d = MvNormal([0.0, 0.0], [1.0 0.0; 0.0 0.0])   # a singular factor
checkparams(d)                                 # false
isfinite(logdensityof(d, [0.0, 0.0]))          # false
```

The density is total in its argument too: a vector of the wrong length is not an
error.

```julia
isnan(logdensityof(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), [0.0]))  # true
```
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
  A structured factor is not a different measure, only a cheaper one, so it dispatches on
  the field type rather than getting a type of its own. These aliases name the two cases
  the methods below specialize on.

  The mean is bound to the struct's own parameter bound rather than left free. An alias
  with an unbounded parameter is not a strict subtype of the general signature, and the
  specializations would be ambiguous with it instead of narrower.
=#
const DiagMvNormal = MvNormal{<:AbstractVector{<:Number},<:Diagonal}
const IsoMvNormal = MvNormal{<:AbstractVector{<:Number},<:UniformScaling}

#=
  A draw is a `Vector`, whatever container the parameters use: it is built from scalar
  arithmetic over the entries of `μ` and `L`.
=#
function Base.eltype(::Type{MvNormal{V,F}}) where {V,F}
    return Vector{float(promote_type(eltype(V), eltype(F)))}
end

"""
    shapesmatch(d::MvNormal, x) -> Bool

Whether `x` and the factor both line up with `μ`.

A branch on this is a branch on shape, not on a value: shapes are static even under
tracing. A `UniformScaling` takes its dimension from `μ`, so there is nothing to check
but `x`.
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

``\\log \\det L``, the sum of the logs of the diagonal, accumulated in `float(T)`.

Taking `T` from the caller keeps an exact factor from capping the term at `Float64` when
the argument is wider, as `Normal` does with `σ`. The accumulator is `float(T)`, not `T`:
a log is not exact, so an exact `T` would be rebound by the first term anyway, and
`Rational` cannot hold one. `logt` keeps a non-positive diagonal from throwing. The
isotropic case is ``n \\log \\lambda`` and needs no loop.
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
  This one is about differentiability, not speed: indexing a `Diagonal` is already `O(1)`,
  but Zygote's adjoint for it produces a structural cotangent, a `NamedTuple`, that it
  cannot then add back to the `Diagonal`. Reading the stored vector keeps the whole
  log-density inside plain array arithmetic. Every other structured method reaches for the
  same field, so this is the only place the general path leaked through.
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

#=
  Use `&` because traced comparisons cannot drive short-circuit evaluation. The shape
  tests are the exception: a size is an `Int` even under tracing.
=#
function checkparams(d::MvNormal)
    n = length(d.μ)
    (n >= 1) & (size(d.L, 1) == n) & (size(d.L, 2) == n) || return false
    ok = all(isfinite, d.μ)
    for i in 1:n
        # The diagonal is what the solve divides by, so it needs a sign as well.
        Lii = d.L[i, i]
        ok &= isfinite(Lii) & (Lii > zero(Lii))
        for j in 1:(i - 1)
            ok &= isfinite(d.L[i, j])
        end
    end
    return ok
end

#=
  A structured factor has only a diagonal to check, so this skips the `O(n²)` sweep over
  off-diagonal zeros that the general method would do.
=#
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

The inner product of the first `k` entries of row `i` of `L` with the first `k` entries
of `v`, for `k >= 1`.

A scalar loop rather than `dot` of two views: every reverse-mode backend understands
arithmetic, and the loop allocates nothing.
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

The standardized value ``L^{-1}(x - \\mu)``, by forward substitution.

`x` must have the length of `μ`, and `L` must be square; [`logdensityof`](@ref
DensityInterface.logdensityof) checks both before calling this.

The result grows by `vcat` instead of being written into a buffer: reverse-mode
backends reject array mutation, and this runs inside a log-density they have to
differentiate. Dividing by a zero or negative diagonal gives a non-finite result
rather than the `SingularException` a library solve would throw.
"""
function whiten(d::MvNormal, x::AbstractVector{<:Number})
    μ, L = d.μ, d.L
    #=
      Take the first row on its own: it fixes the element type from arithmetic, so no
      promotion rule has to be spelled out, and it leaves `rowdot` with a non-empty
      range on every later row.
    =#
    z = [(x[1] - μ[1]) / L[1, 1]]
    for i in 2:length(μ)
        z = vcat(z, (x[i] - μ[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

#=
  A diagonal factor decouples the coordinates, so the substitution collapses to one
  elementwise division: `O(n)` in one allocation, and a broadcast every reverse-mode
  backend already differentiates.
=#
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

#=
  The result type the promotion invariant asks for. `logdensityof` reads it off its own
  arithmetic; this is for the mismatched-shape path, which has no arithmetic to read.
=#
@inline function _densitytype(d::MvNormal, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    # A mismatched shape has no density to return.
    shapesmatch(d, x) || return convert(_densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    #=
      Float the type before converting `log2π`. Exact parameters at an exact point leave
      `q` exact, and `Rational(log2π)` either overflows or throws. This is the promotion
      the univariate measures get for free from `promote_rule(AbstractIrrational, T)`;
      going through `typeof(q)` instead would cap a `Float32` argument at `Float64`.
    =#
    R = float(typeof(q))
    n = length(d.μ)
    return -(q + n * convert(R, log2π)) / 2 - logdetfactor(d, R)
end

#=
  Draw untracked noise and introduce parameters through arithmetic so pathwise
  gradients need no custom AD rule.
=#
function Base.rand(rng::AbstractRNG, d::MvNormal)
    return unwhiten(d, randn(rng, noisetype(d), length(d.μ)))
end

Statistics.mean(d::MvNormal) = d.μ

Statistics.cov(d::MvNormal) = (F=LowerTriangular(d.L); F * F')

#=
  A structured factor squares to a structured covariance, so say so in the type rather
  than materializing a matrix of zeros.
=#
Statistics.cov(d::DiagMvNormal) = Diagonal(abs2.(d.L.diag))
Statistics.cov(d::IsoMvNormal) = Diagonal(fill(abs2(d.L.λ), length(d.μ)))

#=
  The marginal variances, the diagonal of `cov(d)`, without forming the matrix. `std` is
  elementwise: these are marginals, not a matrix to take a square root of.
=#
Statistics.var(d::MvNormal) = map(i -> sum(j -> abs2(d.L[i, j]), 1:i), 1:length(d.μ))
Statistics.var(d::DiagMvNormal) = abs2.(d.L.diag)
Statistics.var(d::IsoMvNormal) = fill(abs2(d.L.λ), length(d.μ))

Statistics.std(d::MvNormal) = sqrt.(var(d))

function entropy(d::MvNormal)
    logdetL = logdetfactor(d, float(_elparamtype(d)))
    # `one(logdetL)`, not `1`: `log2π + 1` would evaluate the `Irrational` at Float64.
    return length(d.μ) * (log2π + one(logdetL)) / 2 + logdetL
end

# The scalar type the parameters promote to, which is what a summary is computed in.
@inline _elparamtype(d::MvNormal) = promote_type(eltype(d.μ), eltype(d.L))

function Base.show(io::IO, d::MvNormal)
    return print(io, "MvNormal(μ=", d.μ, ", L=", d.L, ")")
end
