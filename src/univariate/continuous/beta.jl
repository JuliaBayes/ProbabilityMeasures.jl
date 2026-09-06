"""
    Beta(α, β)
    Beta()

The beta measure on ``[0, 1]`` with shapes `α` and `β`. Its density is

```math
p(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha, \\beta)}
```

`Beta()` creates the uniform measure on ``[0, 1]`` using `Float64` values.

# Arguments

  - `α::Number`: the shape pulling mass towards one.
  - `β::Number`: the shape pulling mass towards zero.

Both shapes must be finite and positive. The constructor does not check them; use
[`validateparams`](@ref) for user input.

The density at an endpoint follows the shape there: infinite below one, finite at one,
zero above.

# Cost

`logdensityof`, `mean`, `var` and `entropy` are closed form. `cdf`, `ccdf`, `logcdf`,
`logccdf`, `quantile`, `median` and `rand` are not: the tails run a continued fraction
and the quantile iterates Newton's method, each until the terms stop changing the
result, in whatever type they are given. Traced numbers run a fixed number of terms
instead; see [`wrappedconditions`](@ref).

A draw is ``X / (X + Y)`` for independent gamma draws with shapes `α` and `β`, each
the quantile of a uniform noise value, combined in log space so that small shapes do
not underflow. Both shapes enter through arithmetic, so automatic differentiation gives
the exact reparameterization gradient with respect to each.
"""
struct Beta{A<:Number,B<:Number} <: ContinuousUnivariateMeasure
    α::A
    β::B
end

Beta() = Beta(1.0, 1.0)

Base.eltype(::Type{Beta{A,B}}) where {A,B} = float(promote_type(A, B))

function checkparams(d::Beta)
    return isfinite(d.α) & isfinite(d.β) & (d.α > zero(d.α)) & (d.β > zero(d.β))
end

support(::Beta) = UnitInterval()

"""
    valuetype(d::Beta, x)

The floating-point type of a density, tail probability, or quantile at `x`.
"""
@inline function valuetype(d::Beta, x::Number)
    return float(promote_type(typeof(d.α), typeof(d.β), typeof(x)))
end

@inline function DensityInterface.logdensityof(d::Beta, x::Number)
    T = valuetype(d, x)
    α, β, y = convert(T, d.α), convert(T, d.β), convert(T, x)
    value = xlogyt(α - one(T), y) + xlogyt(β - one(T), one(T) - y) - logbetat(α, β)
    return select(insupport(d, y) & checkparams(d), () -> value, () -> convert(T, -Inf))
end

#=
  `X / (X + Y)` for gamma draws `X` and `Y` is `1 / (1 + Y/X)`, and the log quantiles
  give `Y/X` without either draw underflowing. The shapes enter through the quantiles,
  so automatic differentiation follows both.
=#
@inline function Base.rand(rng::AbstractRNG, d::Beta)
    F = noisetype(d)
    α, β = map(float, promote(d.α, d.β))
    valid = checkparams(d)
    # `loggamma` throws for a non-positive shape, so invalid parameters take `NaN`.
    lx = select(valid, () -> gammalogquantile(α, rand(rng, F)), () -> oftype(α, NaN))
    ly = select(valid, () -> gammalogquantile(β, rand(rng, F)), () -> oftype(β, NaN))
    return inv(one(lx) + exp(ly - lx))
end

Statistics.mean(d::Beta) = d.α / (d.α + d.β)

function Statistics.var(d::Beta)
    s = d.α + d.β
    return d.α * d.β / (s^2 * (s + one(s)))
end

function entropy(d::Beta)
    α, β = map(float, promote(d.α, d.β))
    valid = checkparams(d)
    # `digamma` has poles at the non-positive integers, so evaluate it at one instead.
    a = select(valid, () -> α, () -> one(α))
    b = select(valid, () -> β, () -> one(β))
    h =
        logbetat(a, b) - (a - one(a)) * digamma(a) - (b - one(b)) * digamma(b) +
        (a + b - 2 * one(a)) * digamma(a + b)
    return select(valid, () -> h, () -> oftype(h, NaN))
end

#=
  The four distribution functions differ only in the tail they take and in the two
  values they hold at the endpoints. The fraction runs only for valid parameters and an
  interior point; elsewhere the answer is known without computing anything.
=#
@inline function betatail(d::Beta, x::Number, below, above, takelower::Bool)
    T = valuetype(d, x)
    α, β, y = convert(T, d.α), convert(T, d.β), convert(T, x)
    valid = checkparams(d) & !isnan(y)
    inside = (y > zero(T)) & (y < one(T))
    tail = select(
        valid & inside,
        () -> logbetainc(α, β, y, one(T) - y)[takelower ? 1 : 2],
        () -> select(y > zero(T), () -> convert(T, above), () -> convert(T, below)),
    )
    return select(valid, () -> tail, () -> convert(T, NaN))
end

cdf(d::Beta, x::Number) = exp(betatail(d, x, -Inf, 0, true))
ccdf(d::Beta, x::Number) = exp(betatail(d, x, 0, -Inf, false))
logcdf(d::Beta, x::Number) = betatail(d, x, -Inf, 0, true)
logccdf(d::Beta, x::Number) = betatail(d, x, 0, -Inf, false)

# Newton's method converges quadratically, so a starting point good to a few digits
# reaches `BigFloat` precision well inside this bound.
const BETAQUANTILE_MAXITER = 100

# The fixed-length path takes this many steps.
const BETAQUANTILE_STEPS = 10

"""
    betaquantile(a, b, p)

The `p`-quantile of `Beta(a, b)`, for `0 < p < 1`.

Newton's method on the logit of `x` refines a closed-form starting point. The logit
keeps both tails accurate where `x` or `1 - x` would round away. The iteration stops
once a step falls below the square root of the rounding error, since quadratic
convergence puts the next step below the rounding error itself, and then takes one more
step for the derivative, as [`gammaquantile`](@ref) does. Traced numbers take
`BETAQUANTILE_STEPS` steps with no value-driven control flow instead.

Newton solves on whichever tail holds the smaller probability, where the log tail has
slope to descend.
"""
function betaquantile(a::Number, b::Number, p::Number)
    fixedlength(a, p) | wrappedconditions(typeof(b)) && return betaquantile_fixed(a, b, p)
    lower = p <= one(p) / 2
    target = lower ? logt(p) : log1p(-p)
    lb = logbetat(a, b)
    v = betaquantile_start(a, b, p)
    tol = sqrt(eps(basefloat(typeof(p))))
    for _ in 1:BETAQUANTILE_MAXITER
        step = betaquantile_step(a, b, lb, v, lower, target)
        v -= step
        abs(step) <= tol && break
    end
    # One more step brings the derivative to the accuracy the value already has.
    return logistic(v - betaquantile_step(a, b, lb, v, lower, target))
end

function betaquantile_fixed(a::Number, b::Number, p::Number)
    lower = p <= one(p) / 2
    target = select(lower, () -> logt(p), () -> log1pt(-p))
    lb = logbetat(a, b)
    v = betaquantile_start(a, b, p)
    for _ in 1:BETAQUANTILE_STEPS
        v -= betaquantile_step(a, b, lb, v, lower, target)
    end
    return logistic(v)
end

@inline logistic(v::Number) = inv(one(v) + exp(-v))

#=
  The Newton step in `v = logit x`. With `x = logistic(v)` and `y = 1 - x`, the tail's
  derivative in `v` is `p(x) x y`, so the step is the tail residual over that slope,
  computed in log space. The tails run in opposite directions, so their steps have
  opposite signs, and the step is bounded so an early overshoot cannot cost many steps.
  Once `x` or `1 - x` has left the normal floating-point range the quantile is as close
  to the endpoint as the type can say, and the step, which would be `NaN`, is zero.
=#
@inline function betaquantile_step(a, b, lb, v, lower, target)
    x = logistic(v)
    y = logistic(-v)
    lp, lq = logbetainc(a, b, x, y)
    tail = select(lower, () -> lp, () -> lq)
    logslope = xlogyt(a, x) + xlogyt(b, y) - lb
    step = (tail - target) * exp(tail - logslope)
    bounded = min(max(step, -3 * one(step)), 3 * one(step))
    least = oftype(x, floatmin(basefloat(typeof(x))))
    safe = select((x > least) & (y > least), () -> bounded, () -> zero(bounded))
    return select(lower, () -> safe, () -> -safe)
end

#=
  The starting point from Numerical Recipes. With both shapes at least one, a
  transformed normal quantile lands within a few percent. Otherwise the power law
  `I_x ≈ x^a / (a B(a, b))` near zero, or its mirror near one, inverts directly, and the
  two are joined where they meet. Both are formed in log space, so a quantile deep in
  either tail starts at a finite logit.
=#
@inline function betaquantile_start(a::Number, b::Number, p::Number)
    # The formula takes the upper normal quantile, positive below the median.
    z = sqrt2 * erfcinvt(2 * p)
    λ = (z^2 - 3 * one(z)) / 6
    ra, rb = inv(2 * a - one(a)), inv(2 * b - one(b))
    h = 2 * inv(ra + rb)
    # `sqrt` sees a negative argument when a shape is below one, where this branch is
    # not used; `abs` keeps it from throwing.
    w = z * sqrt(abs(λ + h)) / h - (rb - ra) * (λ + 5 * one(λ) / 6 - 2 / (3 * h))
    vnormal = logt(a / b) - 2 * w

    lt = xlogyt(a, a / (a + b)) - logt(a)
    lu = xlogyt(b, b / (a + b)) - logt(b)
    lw = lt + log1p(exp(lu - lt))
    # `log x` from the lower power law, `log(1 - x)` from the upper one.
    llo = (logt(a) + lw + logt(p)) / a
    lhi = (logt(b) + lw + log1pt(-p)) / b
    vlo = llo - log1mexpt(min(llo, zero(llo)))
    vhi = -(lhi - log1mexpt(min(lhi, zero(lhi))))
    vpower = select(logt(p) < lt - lw, () -> vlo, () -> vhi)

    normal = (a >= one(a)) & (b >= one(b)) & isfinite(vnormal)
    return select(normal, () -> vnormal, () -> vpower)
end

function Statistics.quantile(d::Beta, p::Number)
    T = valuetype(d, p)
    α, β, q = convert(T, d.α), convert(T, d.β), convert(T, p)
    # A `NaN` probability fails both comparisons.
    valid = checkparams(d) & (q >= zero(T)) & (q <= one(T))
    inside = (q > zero(T)) & (q < one(T))
    edge = select(q == one(T), () -> one(T), () -> zero(T))
    value = select(valid & inside, () -> betaquantile(α, β, q), () -> edge)
    return select(valid, () -> value, () -> convert(T, NaN))
end

function Base.show(io::IO, d::Beta)
    return print(io, "Beta(α=", d.α, ", β=", d.β, ")")
end
