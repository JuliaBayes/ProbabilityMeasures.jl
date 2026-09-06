#=
  The regularized incomplete beta integral, in log space, with the same two paths as
  `gammainc.jl`: numbers whose comparisons give a `Bool` run the continued fraction
  until its terms stop changing the result, and traced numbers run a fixed number of
  terms with `select` in place of every branch.

  The fraction converges quickly only for `x` below `(a + 1) / (a + b + 2)`. Above it,
  the reflection `I_x(a, b) = 1 - I_{1-x}(b, a)` moves the evaluation to the other
  tail, so each tail is computed directly and the other by complement. Callers pass
  `1 - x` alongside `x`, since they often know it more accurately than the subtraction
  would give.
=#

# The fraction needs a number of terms that grows like the square root of the larger
# shape, about fifty for shapes up to `1000` in `Float64`. This bound covers far more.
const BETAINC_MAXITER = 10_000

# The fixed-length path spends this many terms whatever the shapes.
const BETAINC_TERMS = 256

"""
    logbetainc(a, b, x, y)

`log(I_x(a, b))` and `log(1 - I_x(a, b))`, where ``I`` is the regularized incomplete
beta integral

```math
I_x(a, b) = \\frac{1}{B(a, b)} \\int_0^x t^{a-1} (1-t)^{b-1} \\, \\mathrm{d}t
```

and `y` is `1 - x`, passed separately so that a caller who knows it exactly can keep
the upper tail accurate.

The continued fraction runs until its terms stop changing the result. Traced numbers,
for which [`wrappedconditions`](@ref) is true, take a fixed number of terms with no
value-driven control flow instead.
"""
@inline function logbetainc(a::Number, b::Number, x::Number, y::Number)
    fixedlength(a, x) | wrappedconditions(typeof(b)) && return logbetainc_fixed(a, b, x, y)
    if x * (a + b + 2 * one(a)) < a + one(a)
        lp = logbetainc_direct(a, b, x, y)
        return (lp, log1mexpt(lp))
    else
        lq = logbetainc_direct(b, a, y, x)
        return (log1mexpt(lq), lq)
    end
end

@inline function logbetainc_fixed(a::Number, b::Number, x::Number, y::Number)
    lower = x * (a + b + 2 * one(a)) < a + one(a)
    # Evaluate each orientation with its argument clamped to the side where the
    # fraction converges, so the unselected one stays finite.
    crossover = (a + one(a)) / (a + b + 2 * one(a))
    lp = logbetainc_direct_fixed(a, b, min(x, crossover), max(y, one(a) - crossover))
    lq = logbetainc_direct_fixed(b, a, min(y, one(a) - crossover), max(x, crossover))
    logp = select(lower, () -> lp, () -> log1mexpt(lq))
    logq = select(lower, () -> log1mexpt(lp), () -> lq)
    return (logp, logq)
end

#=
  `I_x(a, b) = x^a (1-x)^b / (a B(a, b)) · CF`, with the continued fraction from
  Numerical Recipes evaluated by Lentz's method. The prefactor is kept in log space, so
  a tail stays readable where the probability itself underflows.
=#
@inline function logbetainc_direct(a::Number, b::Number, x::Number, y::Number)
    front = xlogyt(a, x) + xlogyt(b, y) - logbetat(a, b) - logt(a)
    return front + logt(betainc_cf(a, b, x))
end

@inline function logbetainc_direct_fixed(a::Number, b::Number, x::Number, y::Number)
    front = xlogyt(a, x) + xlogyt(b, y) - logbetat(a, b) - logt(a)
    return front + logt(betainc_cf_fixed(a, b, x))
end

#=
  The even partial numerator is exactly zero once `m` reaches an integer `b`, and the
  fraction terminates. As in `gammaq_cf`, the loop then runs on for a term per bit of
  precision so the derivative with respect to the shapes settles along with the value.
=#
function betainc_cf(a::Number, b::Number, x::Number)
    F = basefloat(typeof(x))
    tol = eps(F)
    tiny = oftype(x, sqrt(floatmin(F)))
    qab, qap, qam = a + b, a + one(a), a - one(a)
    c = one(x)
    d = inv(floortiny(one(x) - qab * x / qap, tiny))
    h = inv(floortiny(one(x) - qab * x / qap, tiny))
    zeroed = 0
    for m in 1:BETAINC_MAXITER
        m2 = 2m
        # The even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        (zeroed == 0) & (abs(aa) < tiny) && (zeroed = m)
        d = inv(floortiny(one(x) + aa * d, tiny))
        c = floortiny(one(x) + aa / c, tiny)
        h *= d * c
        # The odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = inv(floortiny(one(x) + aa * d, tiny))
        c = floortiny(one(x) + aa / c, tiny)
        delta = d * c
        h *= delta
        settled = abs(delta - one(delta)) <= tol
        settled & ((zeroed == 0) | (m >= zeroed + precision(F))) && break
    end
    return h
end

function betainc_cf_fixed(a::Number, b::Number, x::Number)
    F = basefloat(typeof(x))
    tiny = oftype(x, sqrt(floatmin(F)))
    qab, qap, qam = a + b, a + one(a), a - one(a)
    c = one(x)
    d = inv(floortiny(one(x) - qab * x / qap, tiny))
    h = inv(floortiny(one(x) - qab * x / qap, tiny))
    for m in 1:BETAINC_TERMS
        m2 = 2m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = inv(floortiny(one(x) + aa * d, tiny))
        c = floortiny(one(x) + aa / c, tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = inv(floortiny(one(x) + aa * d, tiny))
        c = floortiny(one(x) + aa / c, tiny)
        h *= d * c
    end
    return h
end
