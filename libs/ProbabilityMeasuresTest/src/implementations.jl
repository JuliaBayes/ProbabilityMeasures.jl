# Measures, optional methods, and special cases used by the conformance suite.

const UNIVARIATE_OPTIONALS = (:cdf, :quantile, :mean, :var, :std, :median, :entropy)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Normal [
    Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0)
]

_invalids(::Normal) = (Normal(0.0, -1.0), Normal(0.0, 0.0), Normal(Inf, 1.0))
_exactparams(::Normal) = Normal(0, 1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Exponential [
    Exponential(1.0), Exponential(0.4), Exponential(3.0f0)
]

_invalids(::Exponential) = (Exponential(-1.0), Exponential(0.0), Exponential(Inf))
_exactparams(::Exponential) = Exponential(1)

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

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Categorical [
    Categorical([0.2, 0.3, 0.5]), Categorical([1.0]), Categorical(Float32[0.25, 0.75])
]

function _invalids(::Categorical)
    return (Categorical([-0.5, 1.5]), Categorical([0.0, 0.0]), Categorical(Float64[]))
end
_exactparams(::Categorical) = Categorical([1])

default_testpoints(d::Categorical) = float.(eachindex(d.p))

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
