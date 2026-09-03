"""
    Wishart(ν, L)

The Wishart measure on the symmetric positive-definite ``p``-by-``p`` matrices, with `ν`
degrees of freedom and scale matrix ``S = L L'``. Its density is

```math
p(X) = \\frac{|X|^{(\\nu - p - 1)/2}
              \\exp\\left(-\\frac{1}{2}\\operatorname{tr}(S^{-1}X)\\right)}
             {2^{\\nu p/2}\\, |S|^{\\nu/2}\\, \\Gamma_p(\\nu/2)}
```

where ``\\Gamma_p`` is the multivariate gamma function; see [`logmvgamma`](@ref). The
mean is ``\\nu S``.

`L` is a lower-triangular scale factor, the same convention [`MvNormal`](@ref) uses for
its covariance. If you have a scale matrix `S`, factor it before construction:

```julia
using LinearAlgebra
d = Wishart(5.0, Matrix(cholesky(S).L))
```

`ν` must exceed ``p - 1``, below which the measure has no density. The constructor does
not check its arguments; use [`validateparams`](@ref) for user input.

Only the lower triangles of `L` and of the argument are read, and `logdensityof` returns
a non-finite value for invalid parameters, for an argument of the wrong shape, and for
one that is not positive definite. Like [`MvNormal`](@ref), it allocates.

Sampling uses Bartlett's decomposition, which needs one chi-squared draw per dimension
and so inherits [`Gamma`](@ref)'s sampler. The draw is symmetrized, so it lands exactly
in the support rather than a rounding error away from it.
"""
struct Wishart{N<:Number,M<:AbstractMatrix{<:Number}} <: ContinuousMatrixvariateMeasure
    ν::N
    L::M
end

# Samples are matrices regardless of how the parameters are stored.
function Base.eltype(::Type{Wishart{N,M}}) where {N,M}
    return Matrix{float(promote_type(N, eltype(M)))}
end

"""
    dimension(d::Wishart)

The size of one side of a draw.
"""
@inline dimension(d::Wishart) = size(d.L, 1)

"""
    shapesmatch(d::Wishart, X) -> Bool

Whether `X` and `L` are square matrices of the same size.
"""
@inline function shapesmatch(d::Wishart, X::AbstractMatrix)
    p = dimension(d)
    return (p >= 1) & (size(d.L, 2) == p) & (size(X, 1) == p) & (size(X, 2) == p)
end

function checkparams(d::Wishart)
    p = dimension(d)
    (p >= 1) & (size(d.L, 2) == p) || return false
    # Below `p - 1` degrees of freedom the measure is singular and has no density.
    ok = isfinite(d.ν) & (d.ν > p - 1)
    for i in 1:p
        Lii = d.L[i, i]
        ok &= isfinite(Lii) & (Lii > zero(Lii))
        for j in 1:(i - 1)
            ok &= isfinite(d.L[i, j])
        end
    end
    return ok
end

support(d::Wishart) = PositiveDefiniteMatrices(dimension(d))

"""
    logmvgamma(p, a)

`log(Γ_p(a))`, the logarithm of the multivariate gamma function

```math
\\Gamma_p(a) = \\pi^{p(p-1)/4} \\prod_{j=1}^{p}
               \\Gamma\\left(a + \\frac{1 - j}{2}\\right).
```

The product needs `2a > p - 1`, and returns `NaN` otherwise rather than throwing.
"""
function logmvgamma(p::Integer, a::T) where {T<:Number}
    # `loggamma` throws for a non-positive argument, and the smallest one in the product
    # is `a + (1 - p)/2`, so one test covers all of them.
    valid = 2 * a > p - one(T)
    safe = select(valid, () -> a, () -> convert(T, p))
    acc = convert(T, logπ) * ((p * (p - 1)) // 4)
    for j in 1:p
        acc += loggamma(safe + (one(T) - j) / 2)
    end
    # Copy the loop-assigned value to avoid boxing the closure capture.
    total = acc
    return select(valid, () -> total, () -> convert(T, NaN))
end

"""
    mvdigamma(p, a)

The derivative of [`logmvgamma`](@ref) with respect to `a`, ``\\sum_{j=1}^{p}
\\psi(a + (1 - j)/2)``.
"""
function mvdigamma(p::Integer, a::T) where {T<:Number}
    valid = 2 * a > p - one(T)
    safe = select(valid, () -> a, () -> convert(T, p))
    acc = zero(T)
    for j in 1:p
        acc += digamma(safe + (one(T) - j) / 2)
    end
    # Copy the loop-assigned value to avoid boxing the closure capture.
    total = acc
    return select(valid, () -> total, () -> convert(T, NaN))
end

# Result type of the log-density, promoting the parameters with the argument.
@inline function densitytype(d::Wishart, X::AbstractMatrix)
    return float(promote_type(typeof(d.ν), eltype(d.L), eltype(X)))
end

function DensityInterface.logdensityof(d::Wishart, X::AbstractMatrix{<:Number})
    R = densitytype(d, X)
    shapesmatch(d, X) || return convert(R, NaN)
    p = dimension(d)
    ν = convert(R, d.ν)
    C = cholfactor(X)
    logdetX = 2 * logdetdiag(C, p, R)
    # `tr(S⁻¹X)` is the squared Frobenius norm of `L⁻¹C`, taken one column at a time.
    q = sum(j -> sum(abs2, forwardsolve(d.L, C[:, j])), 1:p)
    return (ν - p - 1) / 2 * logdetX - q / 2 - ν * p / 2 * convert(R, logtwo) -
           ν * logdetdiag(d.L, p, R) - logmvgamma(p, ν / 2)
end

"""
    bartlettfactor(rng, d::Wishart)

A lower-triangular `A` such that ``L A A' L'`` is a draw from `d`.

The strictly lower entries are standard normal and the diagonal holds the square roots
of chi-squared draws whose degrees of freedom count down from `ν`.
"""
function bartlettfactor(rng::AbstractRNG, d::Wishart)
    p = dimension(d)
    offdiagonal = randn(rng, noisetype(d), (p * (p - 1)) ÷ 2)
    # `Gamma(k/2, 2)` is the chi-squared measure with `k` degrees of freedom.
    diagonal = map(i -> sqrt(rand(rng, Gamma((d.ν - i + 1) / 2, 2))), 1:p)
    T = eltype(diagonal)
    # Column `j` holds its strictly lower entries at this offset.
    offset(j) = (j - 1) * p - ((j - 1) * j) ÷ 2 - j
    return map(CartesianIndices((p, p))) do I
        i, j = Tuple(I)
        i < j && return zero(T)
        i == j && return diagonal[i]
        return convert(T, offdiagonal[offset(j) + i])
    end
end

function Base.rand(rng::AbstractRNG, d::Wishart)
    M = LowerTriangular(d.L) * bartlettfactor(rng, d)
    X = M * transpose(M)
    # A product is only symmetric up to rounding, and the support is exact.
    return (X + transpose(X)) / 2
end

"""
    scalematrix(d::Wishart)

The scale matrix ``S = L L'``.
"""
function scalematrix(d::Wishart)
    F = LowerTriangular(d.L)
    return F * F'
end

Statistics.mean(d::Wishart) = d.ν * scalematrix(d)

# `Var(X_ij) = ν (S_ij² + S_ii S_jj)`, in the shape of a draw.
function Statistics.var(d::Wishart)
    S = scalematrix(d)
    s = diag(S)
    return d.ν .* (S .^ 2 .+ s * transpose(s))
end

Statistics.std(d::Wishart) = sqrt.(var(d))

# `Cov(X_ij, X_kl) = ν (S_ik S_jl + S_il S_jk)`, over the entries of a draw in the order
# `vec` takes them.
function Statistics.cov(d::Wishart)
    S = scalematrix(d)
    entries = vec(CartesianIndices((dimension(d), dimension(d))))
    return [
        d.ν * (S[a[1], b[1]] * S[a[2], b[2]] + S[a[1], b[2]] * S[a[2], b[1]]) for
        a in entries, b in entries
    ]
end

function entropy(d::Wishart)
    p = dimension(d)
    R = float(promote_type(typeof(d.ν), eltype(d.L)))
    ν = convert(R, d.ν)
    logdetS = 2 * logdetdiag(d.L, p, R)
    return (p + 1) / 2 * logdetS +
           p * (p + 1) / 2 * convert(R, logtwo) +
           logmvgamma(p, ν / 2) - (ν - p - 1) / 2 * mvdigamma(p, ν / 2) + ν * p / 2
end

function Base.show(io::IO, d::Wishart)
    return print(io, "Wishart(ν=", d.ν, ", L=", d.L, ")")
end
