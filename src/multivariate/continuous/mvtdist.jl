"""
    MvTDist(ν, μ, L)

Multivariate Student's t measure with `ν` degrees of freedom, location `μ` and scale
matrix `L * L'`.

`L` is a lower-triangular scale factor, in the same three forms [`MvNormal`](@ref)
accepts: a matrix, a `LinearAlgebra.Diagonal`, or a `LinearAlgebra.UniformScaling`.
Only the lower triangle of a full matrix is used. The density is

```math
p(x) = \\frac{\\Gamma\\!\\left(\\frac{\\nu+n}{2}\\right)}
             {\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)(\\nu\\pi)^{n/2}
              \\prod_i L_{ii}}
       \\left(1 + \\frac{q}{\\nu}\\right)^{-\\frac{\\nu+n}{2}},
\\qquad q = \\lVert L^{-1}(x - \\mu) \\rVert^2 .
```

`L * L'` is the scale matrix, not the covariance: the covariance is
``\\frac{\\nu}{\\nu-2} L L^\\top`` and exists only for ``\\nu > 2``. Below that `cov`,
`var` and `std` are `Inf` or `NaN`, and `mean` is `NaN` for ``\\nu \\le 1``. As `ν`
grows the measure approaches `MvNormal(μ, L)`.

Sampling draws a spherical direction and a radius from the incomplete beta, so it needs
a plain floating-point type. As with [`TDist`](@ref), a draw carries derivatives in `μ`
and `L` but not in `ν`.

The constructor does not check its arguments; use [`validateparams`](@ref) for user
input. `logdensityof` returns a non-finite value for invalid parameters or an argument
of the wrong length, and allocates a temporary vector.
"""
struct MvTDist{N<:Number,V<:AbstractVector{<:Number},F<:ScaleFactor} <: MvLocationScale
    ν::N
    μ::V
    L::F
end

function Base.eltype(::Type{MvTDist{N,V,F}}) where {N,V,F}
    return Vector{float(promote_type(N, eltype(V), eltype(F)))}
end

Base.size(d::MvTDist) = (length(d.μ),)

function checkparams(d::MvTDist)
    return isfinite(d.ν) & (d.ν > zero(d.ν)) & checkfactor(d.μ, d.L)
end

function DensityInterface.logdensityof(d::MvTDist, x::AbstractVector{<:Number})
    shapesmatch(d, x) || return convert(densitytype(d, x), NaN)
    q = sum(abs2, whiten(d, x))
    n = length(d.μ)
    T = float(promote_type(typeof(q), typeof(d.ν)))
    ν = convert(T, d.ν)
    half = (ν + n) / 2
    return tlognorm(ν, n) - logdetfactor(d, T) - half * log1pt(convert(T, q) / ν)
end

#=
  Draw a direction uniformly on the sphere and a radius from the beta representation of
  the squared distance: `q = ν b / (1 - b)` with `b ~ Beta(n/2, ν/2)`. Both come from
  plain noise, so `μ` and `L` still enter the sample through arithmetic alone.
=#
function Base.rand(rng::AbstractRNG, d::MvTDist)
    T = noisetype(d)
    n = length(d.μ)
    ν = convert(T, d.ν)
    z = randn(rng, T, n)
    u = rand(rng, T)
    b, bc = betaincinv(convert(T, n) / 2, ν / 2, u, one(T) - u)
    return unwhiten(d, z .* sqrt(ν * b / (bc * sum(abs2, z))))
end

# Scale `μ` by one or by `NaN` so the result keeps its container type either way.
function Statistics.mean(d::MvTDist)
    T = eltype(eltype(d))
    ν = convert(T, d.ν)
    exists = select(ν > one(T), () -> one(T), () -> convert(T, NaN))
    return exists .* d.μ
end

"""
    covfactor(d::MvTDist)

The factor ``\\nu / (\\nu - 2)`` relating the scale matrix `L * L'` to the covariance.

It is `Inf` for ``1 < \\nu \\le 2`` and `NaN` at or below one, where no covariance
exists.
"""
function covfactor(d::MvTDist)
    T = eltype(eltype(d))
    ν = convert(T, d.ν)
    heavy = select(ν > one(T), () -> convert(T, Inf), () -> convert(T, NaN))
    return select(ν > 2 * one(T), () -> ν / (ν - 2), () -> heavy)
end

Statistics.cov(d::MvTDist) = covfactor(d) * scalematrix(d)

Statistics.var(d::MvTDist) = covfactor(d) .* scalediag(d)

function entropy(d::MvTDist)
    T = float(_promoted_paramtype(typeof(d)))
    ν, n = convert(T, d.ν), length(d.μ)
    half = (ν + n) / 2
    return half * (digamma(half) - digamma(ν / 2)) - tlognorm(ν, n) + logdetfactor(d, T)
end

function Base.show(io::IO, d::MvTDist)
    return print(io, "MvTDist(ν=", d.ν, ", μ=", d.μ, ", L=", d.L, ")")
end
