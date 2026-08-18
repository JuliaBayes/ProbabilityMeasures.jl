#=
  Per-measure declarations: which optional interface components a measure supports,
  which objects to test it with, and the measure-specific hooks the generic suite
  asks for.

  These live here rather than in ProbabilityMeasures itself only so that Interfaces
  stays out of the main dependency graph. If the compile-time `implements` trait
  becomes useful to downstream packages, move the `@implements` lines into the main
  package; Interfaces.jl is small enough that this would be a reasonable trade.
=#

const UNIVARIATE_OPTIONALS = (:cdf, :quantile, :mean, :var, :std, :median, :entropy)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Normal [
    Normal(0.0, 1.0), Normal(-2.5, 0.5), Normal(3.0f0, 2.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. Invalid scales (negative and
  zero) and a non-finite location are both covered, since they fail differently.
=#
_invalids(::Normal) = (Normal(0.0, -1.0), Normal(0.0, 0.0), Normal(Inf, 1.0))
_exactparams(::Normal) = Normal(0, 1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Exponential [
    Exponential(1.0), Exponential(0.4), Exponential(3.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. A negative and a zero scale
  both fail `checkparams`, and a non-finite scale does too.
=#
_invalids(::Exponential) = (Exponential(-1.0), Exponential(0.0), Exponential(Inf))
_exactparams(::Exponential) = Exponential(1)

@implements MeasureInterface{UNIVARIATE_OPTIONALS} Uniform [
    Uniform(0.0, 1.0), Uniform(-1.0, 2.0), Uniform(0.0f0, 2.0f0)
]

#=
  Hooks used by `test_totality` and `test_genericity`. Reversed, empty and unbounded
  intervals all fail `checkparams` and fail differently.

  The exact instance has to contain the test points of every `Uniform` the suite is
  run against, and a width other than one so that the precision check has a `log` that
  is not identically zero.
=#
_invalids(::Uniform) = (Uniform(1.0, 0.0), Uniform(0.0, 0.0), Uniform(-Inf, 1.0))
_exactparams(::Uniform) = Uniform(-1, 2)

#=
  The log-density jumps at the endpoints, so a finite-difference step wider than the
  distance from the test point to the nearest endpoint sends `test_ad`'s reference to
  `-Inf`. The default quantiles come within 0.001 of the width, which leaves that to
  the step-size heuristic; these keep a margin instead. Nothing is lost by pulling them
  in, since every distribution function here is linear, and `test-uniform.jl` checks
  the endpoints directly.
=#
function default_testpoints(d::Uniform)
    return [float(quantile(d, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
end

#=
  `MvNormal` declares the multivariate summaries. The scalar `mean`, `var`, `std` and
  `median` components, and the distribution function, have no multivariate form; the
  blocks of `test_measure` written around a scalar draw skip it too, and
  `test/test-mvnormal.jl` covers what they would have.
=#
const MULTIVARIATE_OPTIONALS = (:meanvector, :cov, :entropy)

@implements MeasureInterface{MULTIVARIATE_OPTIONALS} MvNormal [
    MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
    MvNormal([1.0, -2.0], [2.0 0.0; 0.5 1.5]),
    MvNormal(Float32[0.0, 1.0], Float32[1.0 0.0; -0.25 0.5]),
    MvNormal([1.0, -2.0], Diagonal([2.0, 1.5])),
    MvNormal([1.0, -2.0], 1.5 * I),
]

#=
  Hooks used by `test_totality` and `test_genericity`. Every instance matches the
  dimension of `d`, since they are evaluated at `d`'s test points.

  A zero and a negative diagonal entry both fail `checkparams` and fail differently: the
  first divides by zero in the solve, the second takes the log of a negative number. A
  non-finite mean fails with the factor intact.
=#
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

#=
  The same three defects, keeping the factor's structure. A structured measure has its own
  `checkparams` and its own whitening, so these have to reach those methods rather than
  the general ones a full matrix would dispatch to.
=#
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

#=
  A factor of two on the diagonal rather than the identity: a unit diagonal has
  `log(1) == 0`, which would let a `Float64` intermediate through the precision check.
=#
function _exactparams(d::MvNormal)
    n = length(d.μ)
    return MvNormal(zeros(Int, n), [i == j ? 2 : Int(i > j) for i in 1:n, j in 1:n])
end

function _exactparams(d::DiagMvNormal)
    n = length(d.μ)
    return MvNormal(zeros(Int, n), Diagonal(fill(2, n)))
end

_exactparams(d::IsoMvNormal) = MvNormal(zeros(Int, length(d.μ)), 2 * I)

#=
  Vectors at fixed radii in whitened coordinates, mapped back through `L`, plus a
  wrongly-shaped argument for the totality checks: `logdensityof` is total in the shape
  of its argument as well as its value.
=#
function default_testpoints(d::MvNormal)
    #=
      The radii take the measure's own precision, as `float(quantile(d, p))` does for a
      univariate measure. A `Float64` point against `Float32` parameters would leave
      ReverseDiff holding two tracked types at once, which its scalar `*` cannot resolve.
    =#
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
