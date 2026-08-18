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
  - `L::AbstractMatrix{<:Number}`: the lower-triangular Cholesky factor of the
    covariance. Only its lower triangle is read, as with `LinearAlgebra.LowerTriangular`.

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
struct MvNormal{V<:AbstractVector{<:Number},F<:AbstractMatrix{<:Number}} <:
       ContinuousMultivariateMeasure
    μ::V
    L::F
end

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`.
=#

#=
  A draw is a `Vector`, whatever container the parameters use: it is built from scalar
  arithmetic over the entries of `μ` and `L`.
=#
function Base.eltype(::Type{MvNormal{V,F}}) where {V,F}
    return Vector{float(promote_type(eltype(V), eltype(F)))}
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

support(d::MvNormal) = RealVectors(length(d.μ))

"""
    rowdot(L, v, i, k)

The inner product of the first `k` entries of row `i` of `L` with the first `k` entries
of `v`, for `k >= 1`.

A scalar loop rather than `dot` of two views: every reverse-mode backend understands
arithmetic, and the loop allocates nothing.
"""
@inline function rowdot(L::AbstractMatrix, v::AbstractVector, i::Integer, k::Integer)
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

"""
    unwhiten(d::MvNormal, z)

The inverse of [`whiten`](@ref): ``\\mu + L z``.
"""
function unwhiten(d::MvNormal, z::AbstractVector{<:Number})
    return map(i -> d.μ[i] + rowdot(d.L, z, i, i), 1:length(d.μ))
end

#=
  The result type the promotion invariant asks for. `logdensityof` reads it off its own
  arithmetic; this is for the mismatched-shape path, which has no arithmetic to read.
=#
@inline function _densitytype(d::MvNormal, x::AbstractVector)
    return float(promote_type(eltype(d.μ), eltype(d.L), eltype(x)))
end

function DensityInterface.logdensityof(d::MvNormal, x::AbstractVector{<:Number})
    n = length(d.μ)
    #=
      A branch on shape, not on a value: shapes are static even under tracing, and a
      mismatched one has no density to return.
    =#
    if (length(x) != n) | (n < 1) | (size(d.L, 1) != n) | (size(d.L, 2) != n)
        return convert(_densitytype(d, x), NaN)
    end
    q = sum(abs2, whiten(d, x))
    #=
      Convert the diagonal to the promoted type before taking its log, as `Normal` does
      with `σ`. An exact factor paired with a `BigFloat` argument would otherwise cap
      this term, and `log2π`, at `Float64`.
    =#
    logdetL = zero(q)
    for i in 1:n
        logdetL += logt(oftype(q, d.L[i, i]))
    end
    return -(q + n * oftype(q, log2π)) / 2 - logdetL
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
  The marginal variances, the diagonal of `cov(d)`, without forming the matrix. `std` is
  elementwise: these are marginals, not a matrix to take a square root of.
=#
Statistics.var(d::MvNormal) = map(i -> sum(j -> abs2(d.L[i, j]), 1:i), 1:length(d.μ))
Statistics.std(d::MvNormal) = sqrt.(var(d))

function entropy(d::MvNormal)
    logdetL = sum(i -> logt(float(d.L[i, i])), 1:length(d.μ))
    # `one(logdetL)`, not `1`: `log2π + 1` would evaluate the `Irrational` at Float64.
    return length(d.μ) * (log2π + one(logdetL)) / 2 + logdetL
end

function Base.show(io::IO, d::MvNormal)
    return print(io, "MvNormal(μ=", d.μ, ", L=", d.L, ")")
end
