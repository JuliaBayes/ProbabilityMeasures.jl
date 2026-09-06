# Measures, optional methods, and special cases used by the conformance suite.

const UNIVARIATE_OPTIONALS = (:cdf, :quantile, :mean, :var, :std, :median, :entropy)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Normal [
    Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0)
]

_invalids(::Normal) = (Normal(0.0, -1.0), Normal(0.0, 0.0), Normal(Inf, 1.0))
_exactparams(::Normal) = Normal(0, 1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} LogNormal [
    LogNormal(0.0, 1.0), LogNormal(-1.0, 0.5), LogNormal(2.0f0, 0.75f0)
]

_invalids(::LogNormal) = (LogNormal(0.0, -1.0), LogNormal(0.0, 0.0), LogNormal(Inf, 1.0))
_exactparams(::LogNormal) = LogNormal(0, 1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Exponential [
    Exponential(1.0), Exponential(0.4), Exponential(3.0f0)
]

_invalids(::Exponential) = (Exponential(-1.0), Exponential(0.0), Exponential(Inf))
_exactparams(::Exponential) = Exponential(1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Weibull [
    Weibull(1.0, 1.0), Weibull(0.75, 2.5), Weibull(2.0f0, 0.5f0)
]

function _invalids(::Weibull)
    return (
        Weibull(-1.0, 1.0),
        Weibull(0.0, 1.0),
        Weibull(1.0, 0.0),
        Weibull(Inf, 1.0),
        Weibull(1.0, Inf),
    )
end

_exactparams(::Weibull) = Weibull(2, 3)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Uniform [
    Uniform(0.0, 1.0), Uniform(-1.0, 2.0), Uniform(0.0f0, 2.0f0)
]

_invalids(::Uniform) = (Uniform(1.0, 0.0), Uniform(0.0, 0.0), Uniform(-Inf, 1.0))
# Include every test point and use a width whose logarithm is not zero.
_exactparams(::Uniform) = Uniform(-1, 2)

# Keep finite-difference steps away from the endpoints, where log-density jumps to
# `-Inf`.
function default_testpoints(d::Uniform)
    return [float(quantile(d, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
end

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Laplace [
    Laplace(0.0, 1.0), Laplace(-2.5, 0.5), Laplace(3.0f0, 2.0f0)
]

_invalids(::Laplace) = (Laplace(0.0, -1.0), Laplace(0.0, 0.0), Laplace(Inf, 1.0))
# Use scale 2 so `log(2b)` is not zero during the precision check.
_exactparams(::Laplace) = Laplace(0, 2)

@implements MeasureInterface{(:cdf, :quantile, :median, :entropy)} Cauchy [
    Cauchy(0.0, 1.0), Cauchy(-2.5, 0.5), Cauchy(3.0f0, 2.0f0)
]

function _invalids(::Cauchy)
    return (Cauchy(0.0, -1.0), Cauchy(0.0, 0.0), Cauchy(0.0, Inf), Cauchy(Inf, 1.0))
end
_exactparams(::Cauchy) = Cauchy(0, 2)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Gamma [
    Gamma(2.0, 1.0), Gamma(0.5, 3.0), Gamma(4.5f0, 0.5f0)
]

function _invalids(::Gamma)
    return (Gamma(-1.0, 1.0), Gamma(0.0, 1.0), Gamma(1.0, -1.0), Gamma(Inf, 1.0))
end

# Use a shape and a scale that leave `loggamma(α)` and `log(θ)` non-zero.
_exactparams(::Gamma) = Gamma(3, 2)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Categorical [
    Categorical([0.2, 0.3, 0.5]), Categorical([1.0]), Categorical(Float32[0.25, 0.75])
]

function _invalids(::Categorical)
    return (Categorical([-0.5, 1.5]), Categorical([0.0, 0.0]), Categorical(Float64[]))
end
_exactparams(::Categorical) = Categorical([1])

default_testpoints(d::Categorical) = float.(eachindex(d.p))

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Bernoulli [
    Bernoulli(0.3), Bernoulli(0.5), Bernoulli(0.75f0)
]

# These must be non-finite at the first test point, zero. A negative `p` remains
# finite there and is tested separately.
_invalids(::Bernoulli) = (Bernoulli(1.5), Bernoulli(Inf), Bernoulli(NaN))

# Zero and one give an infinite log-density, so use an exact rational instead.
_exactparams(::Bernoulli) = Bernoulli(1//2)

default_testpoints(::Bernoulli) = [0.0, 1.0]

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Binomial [
    Binomial(1, 0.5), Binomial(5, 0.3), Binomial(4, 0.6f0)
]

# The trial count sets the support and loop lengths.
_structural(::Binomial) = (:n,)

# Keep the original trial count so the test points remain in the support. A negative
# `p` stays finite at zero and is tested separately.
function _invalids(d::Binomial)
    return (Binomial(-1, 0.5), Binomial(d.n, 1.5), Binomial(d.n, NaN))
end

# Use an exact rational probability while leaving the trial count unchanged.
_exactparams(d::Binomial) = Binomial(d.n, 1//2)

# Use outcomes with distinct CDF values. Once rounding makes the CDF equal one,
# quantile cannot recover later outcomes. Keep the last outcome because quantile maps
# a probability of one to it directly.
function _invertible_outcomes(d)
    cs = [cdf(d, float(k)) for k in 0:(d.n)]
    separated(k) = cs[k + 1] > (k == 0 ? zero(eltype(cs)) : cs[k])
    invertible(k) = k == d.n || (separated(k) && cs[k + 1] < one(eltype(cs)))
    return [float(k) for k in 0:(d.n) if invertible(k)]
end

default_testpoints(d::Binomial) = _invertible_outcomes(d)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} BetaBinomial [
    BetaBinomial(1, 1.0, 1.0), BetaBinomial(5, 2.0, 3.0), BetaBinomial(4, 0.5f0, 1.5f0)
]

# The trial count sets the support and loop lengths.
_structural(::BetaBinomial) = (:n,)

# Keep the original trial count so the test points remain in the support. Every one of
# these is non-finite across the whole support.
function _invalids(d::BetaBinomial)
    return (
        BetaBinomial(-one(d.n), d.α, d.β),
        BetaBinomial(d.n, -d.α, d.β),
        BetaBinomial(d.n, d.α, zero(d.β)),
    )
end

# Use exact integer shapes while leaving the trial count unchanged.
_exactparams(d::BetaBinomial) = BetaBinomial(d.n, 2, 3)

default_testpoints(d::BetaBinomial) = _invertible_outcomes(d)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} BetaBinomialLogit [
    BetaBinomialLogit(1, 0.0, 2.0),
    BetaBinomialLogit(5, 0.4, 4.0),
    BetaBinomialLogit(4, -0.5f0, 3.0f0),
]

# The trial count sets the support and loop lengths.
_structural(::BetaBinomialLogit) = (:n,)

# The last of these overflows `exp(η)`, which leaves one implied shape at zero.
function _invalids(d::BetaBinomialLogit)
    return (
        BetaBinomialLogit(-one(d.n), d.η, d.ϕ),
        BetaBinomialLogit(d.n, d.η, -d.ϕ),
        BetaBinomialLogit(d.n, oftype(d.η, 800), d.ϕ),
    )
end

# Use an exact logit and precision while leaving the trial count unchanged.
_exactparams(d::BetaBinomialLogit) = BetaBinomialLogit(d.n, 1, 4)

default_testpoints(d::BetaBinomialLogit) = _invertible_outcomes(d)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Poisson [
    Poisson(0.5), Poisson(4.0), Poisson(2.5f0)
]

# Finite negative rates need separate tests because their density is finite at zero.
_invalids(::Poisson) = (Poisson(-Inf), Poisson(Inf), Poisson(NaN))

# Exercise the `k * log(λ)` term.
_exactparams(::Poisson) = Poisson(2)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Geometric [
    Geometric(0.3), Geometric(0.5), Geometric(0.6f0)
]

#=
  These must be non-finite at every test point, zero included. A probability above one
  stays finite there, so `test-geometric.jl` covers it separately.
=#
_invalids(::Geometric) = (Geometric(0.0), Geometric(-0.5), Geometric(Inf), Geometric(NaN))

# Zero and one give an infinite log-density, so use an exact rational instead.
_exactparams(::Geometric) = Geometric(1//2)

#=
  Several conformance checks use only the first point, and the `k * log(1 - p)` term
  drops out at zero, so lead with a nonzero outcome. Keep zero as well: it is the one
  point where that term is skipped.
=#
function default_testpoints(d::Geometric)
    T = _elscalar(d)
    ps = (T(1) / 4, T(1) / 2, T(3) / 4, T(95) / 100)
    return unique([one(T); [quantile(d, p) for p in ps]])
end

@implements MeasureInterface{(:meanvector, :cov)} Multinomial [
    Multinomial(5, [0.2, 0.3, 0.5]),
    Multinomial(4, Float32[0.25, 0.25, 0.5]),
    Multinomial(3, [1.0]),
]

_structural(::Multinomial) = (:n,)

function _invalids(d::Multinomial)
    negative = map(pᵢ -> -one(float(pᵢ)), d.p)
    nonfinite = map(pᵢ -> oftype(float(pᵢ), NaN), d.p)
    return (
        Multinomial(-one(d.n), d.p), Multinomial(d.n, negative), Multinomial(d.n, nonfinite)
    )
end

# Dyadic weights keep exact log-densities finite at corners and interior points.
function _exactparams(d::Multinomial)
    k = length(d.p)
    p = [i == 1 ? 1//2^(k - 1) : 1//2^(k - i + 1) for i in 1:k]
    return Multinomial(d.n, p)
end

# Several conformance tests use only the first point, so make it an interior point.
function default_testpoints(d::Multinomial)
    T, k = _elscalar(d), length(d.p)
    spread = [convert(T, fld(d.n + k - i, k)) for i in 1:k]
    corners = [[i == j ? convert(T, d.n) : zero(T) for i in 1:k] for j in 1:k]
    return unique(pushfirst!(corners, spread))
end

function _extremepoints(d::Multinomial)
    k = length(d.p)
    return (
        fill(Inf, k),
        fill(-Inf, k),
        fill(NaN, k),
        fill(floatmax(Float64), k),
        zeros(k),
        zeros(k + 1),
        Float64[],
    )
end

# Optional methods for multivariate measures.
const MULTIVARIATE_OPTIONALS = (:meanvector, :cov, :entropy)

@implements MeasureInterface{MULTIVARIATE_OPTIONALS} MvNormal [
    MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
    MvNormal([1.0, -2.0], [2.0 0.0; 0.5 1.5]),
    MvNormal(Float32[0.0, 1.0], Float32[1.0 0.0; -0.25 0.5]),
    MvNormal([1.0, -2.0], Diagonal([2.0, 1.5])),
    MvNormal([1.0, -2.0], 1.5 * I),
]

# Keep invalid examples the same size as the measure under test.
function _invalids(d::MvNormal)
    n, T = length(d.μ), _elscalar(d)
    singular, flipped = _identity(T, n), _identity(T, n)
    singular[1, 1] = 0
    flipped[1, 1] = -1
    return (
        MvNormal(zeros(T, n), singular),
        MvNormal(zeros(T, n), flipped),
        MvNormal(fill(T(Inf), n), _identity(T, n)),
    )
end

_identity(::Type{T}, n::Int) where {T} = T[i == j for i in 1:n, j in 1:n]

# Keep the diagonal factor so its specialized methods are tested.
function _invalids(d::DiagMvNormal)
    n, T = length(d.μ), _elscalar(d)
    singular, flipped = ones(T, n), ones(T, n)
    singular[1] = 0
    flipped[1] = -1
    return (
        MvNormal(zeros(T, n), Diagonal(singular)),
        MvNormal(zeros(T, n), Diagonal(flipped)),
        MvNormal(fill(T(Inf), n), Diagonal(ones(T, n))),
    )
end

function _invalids(d::IsoMvNormal)
    n, T = length(d.μ), _elscalar(d)
    return (
        MvNormal(zeros(T, n), zero(T) * I),
        MvNormal(zeros(T, n), -one(T) * I),
        MvNormal(fill(T(Inf), n), one(T) * I),
    )
end

# Use a non-unit diagonal so the precision test includes a nonzero logarithm.
function _exactparams(d::MvNormal)
    n = length(d.μ)
    return MvNormal(zeros(Int, n), [i == j ? 2 : Int(i > j) for i in 1:n, j in 1:n])
end

function _exactparams(d::DiagMvNormal)
    n = length(d.μ)
    return MvNormal(zeros(Int, n), Diagonal(fill(2, n)))
end

_exactparams(d::IsoMvNormal) = MvNormal(zeros(Int, length(d.μ)), 2 * I)

# Test fixed distances from the mean.
function default_testpoints(d::MvNormal)
    # Keep every coordinate in the measure's numeric type.
    n, T = length(d.μ), _elscalar(d)
    return [unwhiten(d, [isodd(i) ? T(s) : -T(s) for i in 1:n]) for s in (0.75, 0.0, 2.5)]
end

function _extremepoints(d::MvNormal)
    n = length(d.μ)
    return (
        fill(Inf, n),
        fill(-Inf, n),
        fill(NaN, n),
        fill(floatmax(Float64), n),
        zeros(n),
        zeros(n + 1),
        Float64[],
    )
end
