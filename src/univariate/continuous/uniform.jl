"""
    Uniform(a, b)
    Uniform()

The uniform measure on ``[a, b]``, with density

```math
p(x) = \\frac{1}{b - a}
```

with respect to Lebesgue measure. `Uniform()` gives the standard uniform on
``[0, 1]`` in `Float64`.

# Arguments

  - `a::Number`: the lower endpoint.
  - `b::Number`: the upper endpoint.

The `Number` bound permits numeric wrappers used by AD and tracing systems.

Both endpoints belong to the support, matching Distributions.jl.

Construction does not validate. Invalid parameters produce a non-finite density;
use [`checkparams`](@ref) to validate explicitly.

```julia
checkparams(Uniform(1.0, 0.0))               # false
logdensityof(Uniform(1.0, 0.0), 0.5) == -Inf # true
```
"""
struct Uniform{A<:Number,B<:Number} <: ContinuousUnivariateMeasure
    a::A
    b::B
end

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`.
=#

# `Float64` here is a default, not a constraint. Write `Uniform(0.0f0, 1.0f0)` for Float32.
Uniform() = Uniform(0.0, 1.0)

Base.eltype(::Type{Uniform{A,B}}) where {A,B} = float(promote_type(A, B))

checkparams(d::Uniform) = isfinite(d.a) & isfinite(d.b) & (d.a < d.b)

support(d::Uniform) = RealInterval(d.a, d.b)

"""
    uval(d::Uniform, x)

The position of `x` in the interval as a fraction, ``(x - a)/(b - a)``. Values outside
``[0, 1]`` mean `x` is outside the support.
"""
@inline uval(d::Uniform, x::Number) = (x - d.a) / (d.b - d.a)

"""
    xval(d::Uniform, u)

The inverse of [`uval`](@ref): ``a + (b - a)u``.
"""
@inline xval(d::Uniform, u::Number) = muladd(d.b - d.a, u, d.a)

@inline function DensityInterface.logdensityof(d::Uniform, x::Number)
    #=
      The density is flat, so `uval` enters only through its type. Without it an exact
      width would return `Float64` for a `Float32` argument, breaking the promotion
      invariant.
    =#
    u = uval(d, x)
    w = oftype(u, d.b - d.a)
    return select((x >= d.a) & (x <= d.b), () -> -logt(w), () -> oftype(u, -Inf))
end

#=
  Draw untracked noise and stretch it onto the interval, affine like `Normal`'s
  `xval`, so pathwise gradients need no custom AD rule.
=#
@inline function Base.rand(rng::AbstractRNG, d::Uniform)
    return xval(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Uniform) = (d.a + d.b) / 2
Statistics.median(d::Uniform) = mean(d)
Statistics.var(d::Uniform) = (d.b - d.a)^2 / 12

entropy(d::Uniform) = logt(d.b - d.a)

#=
  `clamp` handles the region outside the interval, and lowers to the same `ifelse` a
  traced `select` does. `ccdf` measures from `b` rather than subtracting `cdf` from
  one, saving a rounding step.

  Both are linear, so `log` of either is as accurate as the probability itself. The
  generic `logcdf` and `logccdf` fallbacks need no rewriting here, unlike the ones
  `Normal` and `Exponential` replace.
=#
function cdf(d::Uniform, x::Number)
    u = uval(d, x)
    return clamp(u, zero(u), one(u))
end

function ccdf(d::Uniform, x::Number)
    c = (d.b - x) / (d.b - d.a)
    return clamp(c, zero(c), one(c))
end

#=
  Total for `p` outside `[0, 1]`, which can arrive from float noise in a `cdf`
  round-trip: the result leaves the interval, but no arithmetic here can throw.
=#
Statistics.quantile(d::Uniform, p::Number) = xval(d, p)

function Base.show(io::IO, d::Uniform)
    return print(io, "Uniform(a=", d.a, ", b=", d.b, ")")
end
