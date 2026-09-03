"""
    Gamma(α, θ)
    Gamma(α)
    Gamma()

The gamma measure on ``[0, \\infty)`` with shape `α` and scale `θ`. Its density is

```math
p(x) = \\frac{x^{\\alpha - 1} e^{-x/\\theta}}{\\Gamma(\\alpha)\\, \\theta^{\\alpha}}
```

The parameterization matches Distributions.jl, so the mean is ``\\alpha\\theta``.
`Gamma(1, θ)` is `Exponential(θ)`, and the density at zero follows the shape: infinite
for `α < 1`, `1/θ` at `α = 1`, and zero above.

`Gamma(α)` sets the scale to one in the type of `α`. `Gamma()` creates the unit-shape,
unit-scale measure using `Float64` values.

# Arguments

  - `α::Number`: the shape.
  - `θ::Number`: the scale.

The constructor does not check its arguments. Invalid parameters give a non-finite
density. Use [`checkparams`](@ref) to check them when needed.

```julia
checkparams(Gamma(-1.0, 1.0))               # false
isnan(logdensityof(Gamma(-1.0, 1.0), 1.0))  # true
```

# Cost

`logdensityof` is closed form. `cdf`, `ccdf`, `logcdf`, `logccdf`, `quantile`, `median`
and `entropy` are not: they sum a series, run a continued fraction, or iterate Newton's
method until the terms stop changing the result, which rules out traced and
device-side evaluation. See [`loggammap`](@ref).

Sampling uses Marsaglia and Tsang's rejection method. The accept step runs on plain
floating-point noise, and the accepted noise enters the draw through arithmetic on `α`
and `θ`, so automatic differentiation follows both parameters. That derivative holds
the accepted noise fixed while the acceptance itself depends on `α`, so it is an
approximation, not the exact reparameterization gradient the inverse-CDF samplers give.
"""
struct Gamma{A<:Number,T<:Number} <: ContinuousUnivariateMeasure
    α::A
    θ::T
end

Gamma(α::Number) = Gamma(α, one(α))
Gamma() = Gamma(1.0, 1.0)

Base.eltype(::Type{Gamma{A,T}}) where {A,T} = float(promote_type(A, T))

function checkparams(d::Gamma)
    return isfinite(d.α) & isfinite(d.θ) & (d.α > zero(d.α)) & (d.θ > zero(d.θ))
end

support(::Gamma) = NonNegativeReals()

"""
    valuetype(d::Gamma, x)

The floating-point type of a density, tail probability, or quantile at `x`.

It promotes the parameter types with the type of `x`, so exact parameters keep the
argument's precision.
"""
@inline function valuetype(d::Gamma, x::Number)
    return float(promote_type(typeof(d.α), typeof(d.θ), typeof(x)))
end

@inline function DensityInterface.logdensityof(d::Gamma, x::Number)
    T = valuetype(d, x)
    α, θ, y = convert(T, d.α), convert(T, d.θ), convert(T, x)
    # `loggamma` throws for a non-positive argument, so an invalid shape takes `NaN`.
    lg = select(α > zero(T), () -> loggamma(α), () -> convert(T, NaN))
    value = muladd(α - one(T), logt(y), -(y / θ)) - lg - α * logt(θ)
    # At `x = 0` the formula settles every shape but `α = 1`, where `0 * log(0)` is `NaN`.
    atzero = select(α == one(T), () -> -logt(θ), () -> value)
    return select(
        insupport(d, y) & (y > zero(T)),
        () -> value,
        () -> select(y == zero(T), () -> atzero, () -> convert(T, -Inf)),
    )
end

"""
    gammanoise(rng, a)

The standard normal draw that Marsaglia and Tsang's method accepts for shape `a >= 1`.

``(a - 1/3)(1 + z/\\sqrt{9a - 3})^3`` is then a draw from the unit-scale gamma measure.
The loop does not terminate for a non-finite `a`, so callers must check the shape.
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
    # Promote first so mixed parameter types give a draw of `eltype(d)`.
    α, θ = promote(d.α, d.θ)
    boost = basevalue(α) < one(F)
    a = boost ? α + one(α) : α
    # Invalid parameters would loop forever, so they take `NaN` instead.
    z = checkparams(d) ? gammanoise(rng, convert(F, basevalue(a))) : convert(F, NaN)
    #=
      Apply the parameters after drawing noise so automatic differentiation can follow
      them. This must repeat `gammanoise`'s arithmetic to land on the accepted draw. The
      `abs` keeps the square root from a domain error for an invalid shape, where `z` is
      already `NaN`.
    =#
    c = a - one(a) / 3
    x = θ * c * muladd(inv(sqrt(9 * abs(c))), z, one(c))^3
    return boost ? x * rand(rng, F)^inv(α) : x
end

Statistics.mean(d::Gamma) = d.α * d.θ
Statistics.var(d::Gamma) = d.α * d.θ^2

function entropy(d::Gamma)
    α, θ = map(float, promote(d.α, d.θ))
    # `loggamma` and `digamma` throw for a non-positive argument.
    shape = select(
        α > zero(α), () -> loggamma(α) + (one(α) - α) * digamma(α), () -> oftype(α, NaN)
    )
    return α + logt(θ) + shape
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
    gammaquantile(a, p)

The `p`-quantile of the unit-scale gamma measure with shape `a`, for `0 < p < 1`.

Newton's method on `log(x)` refines a closed-form starting point until the step stops
moving it. The logarithm is what keeps the deep lower tail, where the quantile itself
underflows, from collapsing onto zero at the first step.
"""
function gammaquantile(a::T, p::T) where {T<:Number}
    #=
      Solve on whichever tail holds the smaller probability. The other tail is near one,
      where its logarithm is flat and Newton's method has almost no slope to descend.
      `1 - p` is exact above one half, so the split costs no precision.
    =#
    lower = p <= one(T) / 2
    target = lower ? logt(p) : log1p(-p)

    u = gammaquantile_start(a, p, lower)
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

@inline function gammaquantile_start(a::T, p::T, lower::Bool) where {T<:Number}
    # Small probabilities follow `P(a, y) ≈ y^a / Γ(a+1)`, which inverts directly.
    logr = (logt(p) + loggamma(a + one(T))) / a
    lower && logr < logt((one(T) + a) / 5) && return logr
    # Elsewhere, Wilson and Hilferty's cube-root normal approximation.
    z = -(sqrt2 * erfcinvt(2 * p))
    w = a * (one(T) - inv(9 * a) + z / (3 * sqrt(a)))^3
    return (isfinite(w) & (w > zero(T))) ? logt(w) : logr
end

function Statistics.quantile(d::Gamma, p::Number)
    T = valuetype(d, p)
    checkparams(d) || return convert(T, NaN)
    q = convert(T, p)
    (isnan(q) | (q < zero(T)) | (q > one(T))) && return convert(T, NaN)
    iszero(q) && return zero(T)
    isone(q) && return convert(T, Inf)
    return convert(T, d.θ) * gammaquantile(convert(T, d.α), q)
end

function Base.show(io::IO, d::Gamma)
    return print(io, "Gamma(α=", d.α, ", θ=", d.θ, ")")
end
