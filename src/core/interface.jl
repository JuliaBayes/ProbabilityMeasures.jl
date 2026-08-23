DensityInterface.DensityKind(::AbstractProbabilityMeasure) = DensityInterface.HasDensity()

"""
    params(d) -> NamedTuple

Return the fields of `d` as a named tuple.
"""
@inline function StatsAPI.params(d::D) where {D<:AbstractProbabilityMeasure}
    return NamedTuple{fieldnames(D)}(ntuple(i -> getfield(d, i), Val(fieldcount(D))))
end

"""
    checkparams(d) -> Bool

Return whether `d` has valid parameters.

Constructors do not validate. Invalid parameters make `logdensityof` return `-Inf` or
`NaN` rather than throwing, so use `checkparams` instead of checking for one specific
non-finite value.

# Examples

```julia
d = Normal(0.0, -1.0)     # construction does not throw
checkparams(d)            # false
logdensityof(d, 0.0)      # NaN, not a DomainError
```
"""
checkparams(::AbstractProbabilityMeasure) = true

"""
    validateparams(d) -> d

Return `d`, or throw a `DomainError` when [`checkparams`](@ref) rejects it.

Call this once when user input enters a program, not inside a model or GPU kernel.
This is especially important when invalid parameters can still produce finite
densities, such as categorical probabilities that do not sum to one.

# Examples

```julia
validateparams(Normal(0.0, 1.0))          # returns the measure
validateparams(Categorical([2.0, 2.0]))   # DomainError: does not sum to one
```
"""
function validateparams(d::AbstractProbabilityMeasure)
    checkparams(d) && return d
    throw(DomainError(d, "invalid parameters; see `checkparams`"))
end

"""
    noisetype(d) -> Type{<:AbstractFloat}

The plain floating-point type used to draw random noise for `d`.
"""
@inline function noisetype(d::D) where {D<:AbstractProbabilityMeasure}
    return basefloat(_promoted_paramtype(D))
end

@inline function _promoted_paramtype(::Type{D}) where {D<:AbstractProbabilityMeasure}
    return promote_type(ntuple(i -> eltype(fieldtype(D, i)), Val(fieldcount(D)))...)
end

"""
    masstype(d, x)

The floating-point type for the probability of `x` under a discrete `d`.

It promotes the parameter types with the type of `x`, so a `BigFloat` argument keeps
its precision. A multivariate `x` contributes its element type.
"""
@inline function masstype(::D, x::Number) where {D<:DiscreteMeasure}
    return float(promote_type(_promoted_paramtype(D), typeof(x)))
end

@inline function masstype(::D, x::AbstractVector{<:Number}) where {D<:DiscreteMeasure}
    return float(promote_type(_promoted_paramtype(D), eltype(x)))
end

function Random.rand(
    rng::AbstractRNG, sp::Random.SamplerTrivial{<:AbstractProbabilityMeasure}
)
    return rand(rng, sp[])
end

Base.rand(d::AbstractProbabilityMeasure) = rand(Random.default_rng(), d)

"""
    entropy(d)

The differential (or Shannon) entropy of `d`, in nats.
"""
function entropy end

"""
    cdf(d, x)

``P(X \\le x)`` for ``X \\sim d``.
"""
function cdf end

"""
    ccdf(d, x)

``P(X > x) = 1 - ``[`cdf`](@ref)`(d, x)`, computed so that it stays accurate in the
upper tail.
"""
function ccdf end

"""
    logcdf(d, x)

`log(`[`cdf`](@ref)`(d, x))`, computed so that it stays accurate in the lower tail.
"""
function logcdf end

"""
    logccdf(d, x)

`log(`[`ccdf`](@ref)`(d, x))`, computed so that it stays accurate in the upper tail.
"""
function logccdf end

ccdf(d::UnivariateMeasure, x) = one(cdf(d, x)) - cdf(d, x)
logcdf(d::UnivariateMeasure, x) = logt(cdf(d, x))
logccdf(d::UnivariateMeasure, x) = logt(ccdf(d, x))

# Build one-half in the measure's type so rational parameters do not force `Float64`.
Statistics.median(d::UnivariateMeasure) = quantile(d, one(eltype(d)) / 2)

Statistics.std(d::AbstractProbabilityMeasure) = sqrt(var(d))
