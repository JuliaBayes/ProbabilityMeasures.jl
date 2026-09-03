"""
    Gamma(α, θ)
    Gamma(α)

The gamma measure on ``(0, \\infty)`` with shape `α` and scale `θ`. Its density is

```math
p(x) = \\frac{x^{\\alpha - 1} e^{-x/\\theta}}{\\Gamma(\\alpha)\\, \\theta^{\\alpha}}
```

The mean is ``\\alpha\\theta``, matching Distributions.jl. `Gamma(α)` sets the scale to
one. `Gamma(1, θ)` is `Exponential(θ)`.

# Arguments

  - `α::Number`: the shape.
  - `θ::Number`: the scale.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Gamma(-1.0, 1.0))               # false
isnan(logdensityof(Gamma(-1.0, 1.0), 1.0))  # true
```

`logdensityof` is closed form, so it broadcasts on device arrays and traces. `cdf`,
`ccdf`, `logcdf`, `logccdf`, `quantile`, `median` and `entropy` have no closed form and
iterate until their terms stop changing the result, which rules out traced and
device-side evaluation; see [`loggammap`](@ref).

Sampling uses Marsaglia and Tsang's rejection method. The accept step runs on plain
floating-point noise, so it costs the same whatever numeric type the parameters carry,
and the accepted noise enters the draw through arithmetic on `α` and `θ`, leaving the
draw differentiable with respect to both.
"""
struct Gamma{A<:Number,T<:Number} <: ContinuousUnivariateMeasure
    α::A
    θ::T
end

Gamma(α::Number) = Gamma(α, one(α))

Base.eltype(::Type{Gamma{A,T}}) where {A,T} = float(promote_type(A, T))

function checkparams(d::Gamma)
    return isfinite(d.α) & (d.α > zero(d.α)) & isfinite(d.θ) & (d.θ > zero(d.θ))
end

support(::Gamma) = PositiveReals()

@inline function DensityInterface.logdensityof(d::Gamma, x::Number)
    T = valuetype(d, x)
    α, θ, y = convert(T, d.α), convert(T, d.θ), convert(T, x)
    # `loggamma` throws for a non-positive argument, so an invalid shape takes `NaN`.
    lg = select(α > zero(T), () -> loggamma(α), () -> convert(T, NaN))
    v = muladd(α - one(T), logt(y), -(y / θ)) - lg - α * logt(θ)
    # Convert exact values to a float before returning `-Inf`.
    return select(insupport(d, y), () -> v, () -> convert(T, -Inf))
end

"""
    gammanoise(rng, a)

The standard normal draw that Marsaglia and Tsang's method accepts for shape `a >= 1`.

``(a - 1/3)(1 + z/\\sqrt{9a - 3})^3`` is then a draw from the unit-scale gamma measure.
"""
function gammanoise(rng::AbstractRNG, a::F) where {F<:AbstractFloat}
    c = a - one(F) / 3
    w = inv(sqrt(9 * c))
    while true
        z = randn(rng, F)
        v = muladd(w, z, one(F))
        if v > zero(F)
            u = rand(rng, F)
            v3 = v^3
            # The squeeze accepts most draws without reaching the logarithms.
            if u < muladd(-F(0.0331), z^4, one(F)) ||
                log(u) < z^2 / 2 + c * (one(F) - v3 + log(v3))
                return z
            end
        end
    end
end

#=
  Marsaglia and Tsang's method needs a shape of at least one. Below that, the identity
  `Gamma(α, θ) = Gamma(α + 1, θ) · U^(1/α)` with `U` uniform supplies the rest.
=#
@inline function Base.rand(rng::AbstractRNG, d::Gamma)
    F = noisetype(d)
    boost = basevalue(d.α) < one(F)
    α = boost ? d.α + one(d.α) : d.α
    z = gammanoise(rng, convert(F, basevalue(α)))
    # Apply the parameters after drawing noise so automatic differentiation can follow
    # them. This must repeat `gammanoise`'s arithmetic to land on the accepted draw.
    c = α - one(α) / 3
    x = d.θ * c * muladd(inv(sqrt(9 * c)), z, one(c))^3
    return boost ? x * rand(rng, F)^inv(d.α) : x
end

Statistics.mean(d::Gamma) = d.α * d.θ
Statistics.var(d::Gamma) = d.α * d.θ^2

function entropy(d::Gamma)
    α = float(d.α)
    # `loggamma` and `digamma` throw for a non-positive argument.
    shape = select(
        α > zero(α), () -> loggamma(α) + (one(α) - α) * digamma(α), () -> oftype(α, NaN)
    )
    return α + logt(float(d.θ)) + shape
end

#=
  The four distribution functions differ only in the tail they take and in the two
  values they hold outside `(0, ∞)`, where the answer is known without computing
  anything.
=#
@inline function gammatail(tail, d::Gamma, x::Number, below, above)
    T = valuetype(d, x)
    checkparams(d) || return convert(T, NaN)
    y = convert(T, x) / convert(T, d.θ)
    isnan(y) && return convert(T, NaN)
    y > zero(T) || return convert(T, below)
    isfinite(y) || return convert(T, above)
    return tail(convert(T, d.α), y)
end

cdf(d::Gamma, x::Number) = gammatail((a, y) -> exp(loggammap(a, y)), d, x, 0, 1)
ccdf(d::Gamma, x::Number) = gammatail((a, y) -> exp(loggammaq(a, y)), d, x, 1, 0)
logcdf(d::Gamma, x::Number) = gammatail(loggammap, d, x, -Inf, 0)
logccdf(d::Gamma, x::Number) = gammatail(loggammaq, d, x, 0, -Inf)

# Newton's method converges quadratically, so a starting point good to a few digits
# reaches `BigFloat` precision well inside this bound.
const GAMMAQUANTILE_MAXITER = 100

"""
    gammaquantile(a, p, islower)

The point `y` where the unit-scale gamma measure with shape `a` puts probability `p` on
one tail, for `0 < p < 1`. `islower` selects the tail: `true` solves
[`loggammap`](@ref)`(a, y) == log(p)` and `false` solves [`loggammaq`](@ref).

Taking the tail rather than a lower-tail probability is what lets a measure built on the
upper tail, such as [`InverseGamma`](@ref), invert it without first forming `1 - p` and
losing the small tail to rounding.

Newton's method on `log(y)` refines a closed-form starting point until the step stops
moving it. The logarithm is what keeps the deep lower tail, where the quantile itself
underflows, from collapsing onto zero at the first step.
"""
function gammaquantile(a::T, p::T, islower::Bool) where {T<:Number}
    #=
      Solve on whichever tail holds the smaller probability. The other one is near one,
      where its logarithm is flat and Newton's method has almost no slope to descend.
      `1 - p` is exact above one half, so the switch costs no precision.
    =#
    small = p <= one(T) / 2
    lower = islower == small
    target = small ? logt(p) : log1p(-p)

    u = gammaquantile_start(a, target, lower)
    tol = eps(basefloat(T))
    for _ in 1:GAMMAQUANTILE_MAXITER
        y = exp(u)
        (isfinite(y) & (y > zero(T))) || break
        # The log-density of the unit-scale measure, written in `u` so that it stays
        # accurate where `y` underflows.
        g = muladd(a - one(T), u, -y) - loggamma(a)
        tail = lower ? loggammap(a, y) : loggammaq(a, y)
        # The tails run in opposite directions, so their Newton steps do too.
        step = (tail - target) * exp(tail - g - u)
        u = lower ? u - step : u + step
        abs(step) <= tol * max(abs(u), one(T)) && break
    end
    return exp(u)
end

# `target` is the logarithm of the tail being solved, which always holds at most half
# the mass, so the lower-tail probability is `target` itself or its complement.
@inline function gammaquantile_start(a::T, target::T, lower::Bool) where {T<:Number}
    tail = exp(target)
    loglower = lower ? target : log1p(-tail)
    # A small lower tail follows `P(a, y) ≈ y^a / Γ(a+1)`, which inverts directly.
    logr = (loglower + loggamma(a + one(T))) / a
    logr < logt((one(T) + a) / 5) && return logr
    # Elsewhere, Wilson and Hilferty's cube-root normal approximation. The normal
    # quantile of the tail being solved changes sign with the tail.
    z = sqrt2 * erfcinvt(2 * tail)
    w = a * (one(T) - inv(9 * a) + (lower ? -z : z) / (3 * sqrt(a)))^3
    return (isfinite(w) & (w > zero(T))) ? logt(w) : logr
end

function Statistics.quantile(d::Gamma, p::Number)
    T = valuetype(d, p)
    checkparams(d) || return convert(T, NaN)
    q = convert(T, p)
    (isnan(q) | (q < zero(T)) | (q > one(T))) && return convert(T, NaN)
    iszero(q) && return zero(T)
    isone(q) && return convert(T, Inf)
    return convert(T, d.θ) * gammaquantile(convert(T, d.α), q, true)
end

function Base.show(io::IO, d::Gamma)
    return print(io, "Gamma(α=", d.α, ", θ=", d.θ, ")")
end
