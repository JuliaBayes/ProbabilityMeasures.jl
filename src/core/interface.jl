# The generic surface every measure inherits. Anything here either has a correct
# fallback or is documented as mandatory in `AbstractProbabilityMeasure`.

DensityInterface.DensityKind(::AbstractProbabilityMeasure) = DensityInterface.HasDensity()

"""
    params(d) -> NamedTuple

The parameters of `d`, keyed by field name.

A `NamedTuple` rather than a positional tuple: a PPL needs to address parameters by
name to build traces and priors, and it costs nothing at runtime. The default reads
the fields directly, so measures rarely define this.
"""
@inline function StatsAPI.params(d::D) where {D<:AbstractProbabilityMeasure}
    # `D` is a type parameter, so `fieldnames`/`fieldcount` fold to constants and
    # this compiles to a plain struct load.
    return NamedTuple{fieldnames(D)}(ntuple(i -> getfield(d, i), Val(fieldcount(D))))
end

"""
    checkparams(d) -> Bool

Whether `d`'s parameters are valid (e.g. a positive scale).

**Constructors in this package never validate.** That is a deliberate break from
Distributions.jl, for two reasons: a constructor that can `throw` cannot be called
from inside a GPU kernel, and a PPL constructs measures in its innermost loop where
a branch and an error path are not free. Validation is opt-in, at the boundaries
where a human supplied the numbers.

Invalid parameters are not silently wrong -- by invariant 2 of
[`AbstractProbabilityMeasure`](@ref), `logdensityof` returns `NaN` for them.

```julia
d = Normal(0.0, -1.0)     # constructs fine, no error
checkparams(d)            # false
logdensityof(d, 0.0)      # NaN, not a DomainError
```
"""
checkparams(::AbstractProbabilityMeasure) = true

"""
    noisetype(d) -> Type{<:AbstractFloat}

The plain float type in which to draw the underlying randomness for a
reparameterized sample from `d`.

See [`basefloat`](@ref) for why the AD tracking is stripped here.
"""
@inline function noisetype(d::D) where {D<:AbstractProbabilityMeasure}
    return basefloat(_promoted_paramtype(D))
end

@inline function _promoted_paramtype(::Type{D}) where {D<:AbstractProbabilityMeasure}
    return promote_type(ntuple(i -> fieldtype(D, i), Val(fieldcount(D)))...)
end

# `rand(d, n)`, `rand(rng, d, dims...)` and `rand!(A, d)` all route through Random's
# sampler machinery. Measures implement the scalar `Base.rand(rng, d)`; this hands
# the array forms back to it.
function Random.rand(
    rng::AbstractRNG, sp::Random.SamplerTrivial{<:AbstractProbabilityMeasure}
)
    return rand(rng, sp[])
end

Base.rand(d::AbstractProbabilityMeasure) = rand(Random.default_rng(), d)

# --- Moments and summaries ------------------------------------------------------
#
# `mean`, `var`, `std`, `median` and `quantile` are extended from Statistics; the
# rest live in StatsBase, which is far too heavy a dependency for what amounts to
# five function names, so they are defined here.

"""
    mode(d)

The location of the maximum of the density of `d`.
"""
function mode end

"""
    entropy(d)

The differential (or Shannon) entropy of `d`, in nats.
"""
function entropy end

"""
    skewness(d)

The standardized third central moment of `d`.
"""
function skewness end

"""
    kurtosis(d)

The *excess* kurtosis of `d`: the standardized fourth central moment minus three,
so that a normal measure has zero kurtosis.
"""
function kurtosis end

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

"""
    mgf(d, t)

The moment generating function ``E[e^{tX}]``.
"""
function mgf end

"""
    cf(d, t)

The characteristic function ``E[e^{itX}]``.
"""
function cf end

# Generic fallbacks. These are correct for any measure that defines the primitive
# they delegate to, and each is accurate over the range where the primitive is.
ccdf(d::UnivariateMeasure, x) = one(cdf(d, x)) - cdf(d, x)
logcdf(d::UnivariateMeasure, x) = logt(cdf(d, x))
logccdf(d::UnivariateMeasure, x) = logt(ccdf(d, x))
Statistics.median(d::UnivariateMeasure) = quantile(d, 1//2)
Statistics.std(d::AbstractProbabilityMeasure) = sqrtt(var(d))
