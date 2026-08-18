#=
  The generic surface every measure inherits. Anything here either has a correct
  fallback or is documented as mandatory in `AbstractProbabilityMeasure`.
=#

DensityInterface.DensityKind(::AbstractProbabilityMeasure) = DensityInterface.HasDensity()

"""
    params(d) -> NamedTuple

The parameters of `d`, keyed by field name.

The default returns the fields as a `NamedTuple`, allowing callers to address
parameters by name.
"""
@inline function StatsAPI.params(d::D) where {D<:AbstractProbabilityMeasure}
    # `D` is a type parameter, so this folds to constants and compiles to a struct load.
    return NamedTuple{fieldnames(D)}(ntuple(i -> getfield(d, i), Val(fieldcount(D))))
end

"""
    checkparams(d) -> Bool

Whether `d`'s parameters are valid, for example a positive scale.

Constructors in this package do not validate. Validation is explicit so construction
can remain usable inside compiled kernels and PPL inner loops.

For invalid parameters, `logdensityof` returns a non-finite value rather than
throwing.

Use this function rather than `isnan(logdensityof(d, x))`: an invalid measure may
produce `-Inf` rather than `NaN`.

# Examples

```julia
d = Normal(0.0, -1.0)     # constructs fine, no error
checkparams(d)            # false
logdensityof(d, 0.0)      # NaN, not a DomainError
```
"""
checkparams(::AbstractProbabilityMeasure) = true

"""
    noisetype(d) -> Type{<:AbstractFloat}

The untracked float type used to draw noise for a reparameterized sample from `d`.
"""
@inline function noisetype(d::D) where {D<:AbstractProbabilityMeasure}
    return basefloat(_promoted_paramtype(D))
end

#=
  `eltype` of a number type is that type, so this folds array parameters in on their
  element type and leaves scalar parameters alone.
=#
@inline function _promoted_paramtype(::Type{D}) where {D<:AbstractProbabilityMeasure}
    return promote_type(ntuple(i -> eltype(fieldtype(D, i)), Val(fieldcount(D)))...)
end

#=
  Array sampling routes through Random's sampler machinery to the scalar
  `Base.rand(rng, d)` implementation.
=#
function Random.rand(
    rng::AbstractRNG, sp::Random.SamplerTrivial{<:AbstractProbabilityMeasure}
)
    return rand(rng, sp[])
end

Base.rand(d::AbstractProbabilityMeasure) = rand(Random.default_rng(), d)

#=
  Extend Statistics for standard summaries; `entropy` is the only package-specific
  summary currently needed by PPL workloads.
=#

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

#=
  Generic fallbacks for measures that define the corresponding primitive.
=#
ccdf(d::UnivariateMeasure, x) = one(cdf(d, x)) - cdf(d, x)
logcdf(d::UnivariateMeasure, x) = logt(cdf(d, x))
logccdf(d::UnivariateMeasure, x) = logt(ccdf(d, x))

#=
  Construct one-half in `eltype(d)`; `float(::Rational)` would force `Float64`.
=#
Statistics.median(d::UnivariateMeasure) = quantile(d, one(eltype(d)) / 2)

#=
  A conforming measure has non-negative variance, so plain `sqrt` is total here.
=#
Statistics.std(d::AbstractProbabilityMeasure) = sqrt(var(d))
