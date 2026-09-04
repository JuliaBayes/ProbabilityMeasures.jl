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

`logdensityof` is closed form. `cdf`, `ccdf`, `logcdf`, `logccdf`, `quantile`, `median`,
`entropy` and `rand` are not: they sum a series, run a continued fraction, or iterate
Newton's method. Each loop runs until its terms stop changing the result, in the type
it is given, so `BigFloat` keeps its precision and differentiation tools follow the
iteration. Traced numbers, for which [`wrappedconditions`](@ref) is true, run a fixed
number of terms with no value-driven control flow instead, so the same code compiles
under Reactant; that path is exact for shapes up to about `900` in `Float64` and `2000`
in `Float32`, and returns `NaN` above. See [`loggammap`](@ref).

Sampling inverts the CDF: a draw is `θ` times the unit-scale quantile at a uniform noise
value. Both parameters enter through arithmetic, so automatic differentiation gives the
exact reparameterization gradient with respect to each. The price is a Newton solve per
draw, a few microseconds at moderate shapes.
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

# The noise is plain, and the parameters enter through `quantile`, so automatic
# differentiation follows them through the Newton solve.
@inline function Base.rand(rng::AbstractRNG, d::Gamma)
    return quantile(d, rand(rng, noisetype(d)))
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
  values they hold at the ends of the support, where the answer is known without
  computing anything. `tail` runs only for valid parameters, since `loggamma` throws
  for a non-positive shape.
=#
@inline function gammatail(tail, d::Gamma, x::Number, below, above)
    T = valuetype(d, x)
    α, s = convert(T, d.α), convert(T, x) / convert(T, d.θ)
    valid = checkparams(d) & !isnan(s)
    inside = isfinite(s) & (s > zero(T))
    edge = select(s > zero(T), () -> convert(T, above), () -> convert(T, below))
    value = select(valid & inside, () -> tail(α, s), () -> edge)
    return select(valid, () -> value, () -> convert(T, NaN))
end

cdf(d::Gamma, x::Number) = gammatail((a, y) -> exp(loggammap(a, y)), d, x, 0, 1)
ccdf(d::Gamma, x::Number) = gammatail((a, y) -> exp(loggammaq(a, y)), d, x, 1, 0)
logcdf(d::Gamma, x::Number) = gammatail(loggammap, d, x, -Inf, 0)
logccdf(d::Gamma, x::Number) = gammatail(loggammaq, d, x, 0, -Inf)

# Newton's method converges quadratically, so a starting point good to a few digits
# reaches `BigFloat` precision well inside this bound.
const GAMMAQUANTILE_MAXITER = 100

# The fixed-length path takes this many steps. The starting point is good to two or
# three digits everywhere, which quadratic convergence carries past `Float64` in four.
const GAMMAQUANTILE_STEPS = 8

"""
    gammaquantile(a, p)

The `p`-quantile of the unit-scale gamma measure with shape `a`, for `0 < p < 1`.

Newton's method on `log(x)` refines a closed-form starting point. The logarithm is what
keeps the deep lower tail, where the quantile itself underflows, from collapsing onto
zero at the first step. The iteration stops once a step falls below the square root of
the rounding error, since quadratic convergence puts the next step below the rounding
error itself. Traced numbers take `GAMMAQUANTILE_STEPS` steps with no value-driven
control flow instead; see [`wrappedconditions`](@ref).

Newton solves on whichever tail holds the smaller probability. The other tail is near
one, where its logarithm is flat and the method has almost no slope to descend. `1 - p`
is exact above one half, so the split costs no precision.
"""
function gammaquantile(a::Number, p::Number)
    fixedlength(a, p) && return gammaquantile_fixed(a, p)
    lower = p <= one(p) / 2
    target = lower ? logt(p) : log1p(-p)
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    tol = sqrt(eps(basefloat(typeof(p))))
    for _ in 1:GAMMAQUANTILE_MAXITER
        y = exp(u)
        (isfinite(y) & (y > zero(y))) || break
        step = gammaquantile_step(a, lg, u, y, lower, target)
        u -= step
        abs(step) <= tol && break
    end
    return exp(u)
end

function gammaquantile_fixed(a::Number, p::Number)
    lower = p <= one(p) / 2
    target = select(lower, () -> logt(p), () -> log1pt(-p))
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    for _ in 1:GAMMAQUANTILE_STEPS
        u -= gammaquantile_step(a, lg, u, exp(u), lower, target)
    end
    return exp(u)
end

#=
  The Newton step in `u = log y`. `g` is the log-density of the unit-scale measure,
  written in `u` so that it stays accurate where `y` is tiny. The tails run in opposite
  directions, so their steps have opposite signs.

  Far from the root the step can overshoot deep into the upper tail, from where Newton
  crawls back one unit of `u` per step, so the step is bounded. A subnormal `y` carries
  too few digits for the tails to settle, and a zero one gives `NaN`; the quantile is
  then below the smallest normal number and the current `u` is as good as it gets.
=#
@inline function gammaquantile_step(a, lg, u, y, lower, target)
    g = muladd(a - one(a), u, -y) - lg
    tail = select(lower, () -> loggammap(a, y), () -> loggammaq(a, y))
    step = (tail - target) * exp(tail - g - u)
    bounded = min(max(step, -2 * one(step)), 2 * one(step))
    normal = y > oftype(y, floatmin(basefloat(typeof(y))))
    safe = select(normal, () -> bounded, () -> zero(bounded))
    return select(lower, () -> safe, () -> -safe)
end

#=
  Near zero, `P(a, y) ≈ y^a / Γ(a+1)` inverts directly. Its relative error is about
  `a y / (a + 1)`, so it serves while `y` stays below `(a + 1) / (5a)`, which for a small
  shape covers most of the mass, upper tail included. Elsewhere, Wilson and Hilferty's
  cube-root normal approximation is good to a few digits.
=#
@inline function gammaquantile_start(a::Number, p::Number)
    logr = (logt(p) + loggamma(a + one(a))) / a
    z = -(sqrt2 * erfcinvt(2 * p))
    w = a * (one(a) - inv(9 * a) + z / (3 * sqrt(a)))^3
    small = logr < logt((one(a) + a) / (5 * a))
    usable = isfinite(w) & (w > zero(w))
    return select(small | !usable, () -> logr, () -> logt(w))
end

function Statistics.quantile(d::Gamma, p::Number)
    T = valuetype(d, p)
    α, θ, q = convert(T, d.α), convert(T, d.θ), convert(T, p)
    # A `NaN` probability fails both comparisons.
    valid = checkparams(d) & (q >= zero(T)) & (q <= one(T))
    inside = (q > zero(T)) & (q < one(T))
    edge = select(q == one(T), () -> convert(T, Inf), () -> zero(T))
    value = select(valid & inside, () -> θ * gammaquantile(α, q), () -> edge)
    return select(valid, () -> value, () -> convert(T, NaN))
end

function Base.show(io::IO, d::Gamma)
    return print(io, "Gamma(α=", d.α, ", θ=", d.θ, ")")
end
