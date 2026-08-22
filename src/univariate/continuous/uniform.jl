"""
    Uniform(a, b)
    Uniform()

The uniform measure on ``[a, b]``. Its density is

```math
p(x) = \\frac{1}{b - a}
```

`Uniform()` creates the standard uniform measure on ``[0, 1]`` using `Float64`
values.

# Arguments

  - `a::Number`: the lower endpoint.
  - `b::Number`: the upper endpoint.

Both endpoints belong to the support, matching Distributions.jl.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Uniform(1.0, 0.0))               # false
logdensityof(Uniform(1.0, 0.0), 0.5) == -Inf # true
```
"""
struct Uniform{A<:Number,B<:Number} <: ContinuousUnivariateMeasure
    a::A
    b::B
end

Uniform() = Uniform(0.0, 1.0)

Base.eltype(::Type{Uniform{A,B}}) where {A,B} = float(promote_type(A, B))

checkparams(d::Uniform) = isfinite(d.a) & isfinite(d.b) & (d.a < d.b)

support(d::Uniform) = RealInterval(d.a, d.b)

"""
    uval(d::Uniform, x)

Return the position of `x` in the interval as a fraction, ``(x - a)/(b - a)``.
Values outside ``[0, 1]`` are outside the support.
"""
@inline uval(d::Uniform, x::Number) = (x - d.a) / (d.b - d.a)

"""
    xval(d::Uniform, u)

Convert `u` back to the interval: ``a + (b - a)u``.
"""
@inline xval(d::Uniform, u::Number) = muladd(d.b - d.a, u, d.a)

@inline function DensityInterface.logdensityof(d::Uniform, x::Number)
    # Use `u`'s type so integer endpoints do not override the argument's precision.
    u = uval(d, x)
    w = oftype(u, d.b - d.a)
    # Convert exact values to a float before returning `-Inf`.
    return select((x >= d.a) & (x <= d.b), () -> -logt(w), () -> oftype(float(u), -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::Uniform)
    return xval(d, rand(rng, noisetype(d)))
end

Statistics.mean(d::Uniform) = (d.a + d.b) / 2
Statistics.median(d::Uniform) = mean(d)
Statistics.var(d::Uniform) = (d.b - d.a)^2 / 12

entropy(d::Uniform) = logt(d.b - d.a)

# Compute each tail from its own endpoint instead of subtracting from one. The generic
# log-CDF methods are accurate because both tails are linear.
function cdf(d::Uniform, x::Number)
    u = uval(d, x)
    return clamp(u, zero(u), one(u))
end

function ccdf(d::Uniform, x::Number)
    c = (d.b - x) / (d.b - d.a)
    return clamp(c, zero(c), one(c))
end

# A probability outside `[0, 1]` returns a value outside the interval without throwing.
Statistics.quantile(d::Uniform, p::Number) = xval(d, p)

function Base.show(io::IO, d::Uniform)
    return print(io, "Uniform(a=", d.a, ", b=", d.b, ")")
end
