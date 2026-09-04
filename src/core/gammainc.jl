#=
  The regularized incomplete gamma integrals, in log space. Returning logarithms keeps
  a tail readable where the probability itself underflows, and every step stays in the
  type of the arguments, so `BigFloat` keeps its precision.

  There are two paths, chosen by `wrappedconditions`. Numbers whose comparisons give a
  `Bool` run each expansion until its terms stop changing the result. Traced numbers run
  a fixed number of terms with `select` in place of every value-driven branch, so
  Reactant can compile the loop.

  Types come from the values rather than from a shared type parameter, because the
  numbers a reverse-mode tape produces need not share a type with the ones it was given.
=#

#=
  Both expansions converge geometrically once `x` is on its own side of `a + 1`. At the
  crossover they need about `√(2a log(1/ε))` terms to reach the rounding error `ε`:
  `8.5√a` in `Float64`, more at higher precision. This bound covers `Float64` shapes up
  to roughly `10^6`. Past it the remaining terms are dropped and the result loses
  accuracy.
=#
const GAMMAINC_MAXITER = 10_000

#=
  The fixed-length path spends this many terms on each expansion whatever the shape,
  so it is exact only up to `gammainc_maxshape` and returns `NaN` above it.
=#
const GAMMAINC_TERMS = 256

"""
    gammainc_maxshape(T)

The largest shape for which the fixed-length [`loggammap`](@ref) and
[`loggammaq`](@ref) reach the rounding error of `T` with `GAMMAINC_TERMS` terms: about
`900` for a `Float64` and `2000` for a `Float32` wrapped in `T`.
"""
@inline function gammainc_maxshape(::Type{T}) where {T}
    ε = eps(basefloat(T))
    return GAMMAINC_TERMS^2 / (-2 * log(ε))
end

# Either argument may be the traced one.
@inline fixedlength(a, x) = wrappedconditions(typeof(a)) | wrappedconditions(typeof(x))

"""
    loggammap(a, x)

`log(P(a, x))`, where ``P`` is the lower incomplete gamma integral divided by
``\\Gamma(a)``:

```math
P(a, x) = \\frac{1}{\\Gamma(a)} \\int_0^x t^{a-1} e^{-t} \\, \\mathrm{d}t.
```

`a` must be positive and `x` non-negative. Below `x = a + 1` this sums a series whose
terms are all positive; above it, it takes the complement of [`loggammaq`](@ref).

Each expansion runs until its terms stop changing the result. Traced numbers, for which
[`wrappedconditions`](@ref) is true, take a fixed number of terms with no value-driven
control flow instead; that path is exact up to [`gammainc_maxshape`](@ref) and returns
`NaN` above it.

The prefactor is a difference of terms of order ``a \\log a``, so relative accuracy
holds to the rounding error of the argument type for shapes up to about `1000` and then
falls off roughly in proportion to the shape.

See also [`loggammaq`](@ref).
"""
@inline function loggammap(a::Number, x::Number)
    fixedlength(a, x) && return loggammapq_fixed(a, x)[1]
    return x < a + one(a) ? gammap_series(a, x) : log1mexpt(gammaq_cf(a, x))
end

"""
    loggammaq(a, x)

`log(Q(a, x))`, where ``Q(a, x) = 1 - P(a, x)`` is the upper incomplete gamma integral
divided by ``\\Gamma(a)``.

Above `x = a + 1` this evaluates a continued fraction; below it, it takes the
complement of [`loggammap`](@ref). The two paths are the same as for `loggammap`.
"""
@inline function loggammaq(a::Number, x::Number)
    fixedlength(a, x) && return loggammapq_fixed(a, x)[2]
    return x < a + one(a) ? log1mexpt(gammap_series(a, x)) : gammaq_cf(a, x)
end

"""
    loggammapq_fixed(a, x)

Both `log(P(a, x))` and `log(Q(a, x))` from the fixed-length expansions.

Each expansion is evaluated on its own side of the crossover, with `x` clamped to the
crossover for the other, so neither can overflow before `select` discards it.
"""
@inline function loggammapq_fixed(a::Number, x::Number)
    crossover = a + one(a)
    lower = x < crossover
    ls = gammap_series_fixed(a, min(x, crossover))
    lq = gammaq_cf_fixed(a, max(x, crossover))
    logp = select(lower, () -> ls, () -> log1mexpt(lq))
    logq = select(lower, () -> log1mexpt(ls), () -> lq)
    exact = a <= gammainc_maxshape(typeof(x))
    nan = oftype(ls, NaN)
    return (select(exact, () -> logp, () -> nan), select(exact, () -> logq, () -> nan))
end

#=
  `P(a, x) = x^a e^{-x} / Γ(a+1) · Σₙ xⁿ / ((a+1)⋯(a+n))`. Every term is positive, so
  the sum loses nothing to cancellation.
=#
function gammap_series(a::Number, x::Number)
    tol = eps(basefloat(typeof(x)))
    term = one(x)
    total = one(x)
    n = 0
    while (n < GAMMAINC_MAXITER) & (abs(term) > tol * abs(total))
        n += 1
        term *= x / (a + n)
        total += term
    end
    return muladd(a, logt(x), -x) - loggamma(a + one(a)) + logt(total)
end

function gammap_series_fixed(a::Number, x::Number)
    term = one(x)
    total = one(x)
    for n in 1:GAMMAINC_TERMS
        term *= x / (a + n)
        total += term
    end
    return muladd(a, logt(x), -x) - loggamma(a + one(a)) + logt(total)
end

#=
  `Q(a, x) = x^a e^{-x} / Γ(a) · CF`, with the continued fraction

      CF = 1/(b₀ - a₁/(b₁ - a₂/(b₂ - ⋯))),   b₀ = x + 1 - a, bᵢ = b₀ + 2i, aᵢ = i(i - a)

  evaluated by Lentz's method, which builds the fraction from the front and so needs no
  guess at where to truncate it. `tiny` replaces a denominator that rounds to zero,
  which is what lets the recurrence step past a vanishing partial numerator. Its square
  is still a normal number, so differentiating through the guard gives zero, not `NaN`.
=#
function gammaq_cf(a::Number, x::Number)
    F = basefloat(typeof(x))
    tol = eps(F)
    tiny = oftype(x, sqrt(floatmin(F)))
    b = x + one(x) - a
    c = oftype(x, inv(sqrt(floatmin(F))))
    d = inv(b)
    h = inv(b)
    for i in 1:GAMMAINC_MAXITER
        an = -i * (i - a)
        b += 2 * one(x)
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        delta = d * c
        h *= delta
        abs(delta - one(delta)) <= tol && break
    end
    return muladd(a, logt(x), -x) - loggamma(a) + logt(h)
end

function gammaq_cf_fixed(a::Number, x::Number)
    F = basefloat(typeof(x))
    tiny = oftype(x, sqrt(floatmin(F)))
    b = x + one(x) - a
    c = oftype(x, inv(sqrt(floatmin(F))))
    d = inv(b)
    h = inv(b)
    for i in 1:GAMMAINC_TERMS
        an = -i * (i - a)
        b += 2 * one(x)
        d = inv(floortiny(an * d + b, tiny))
        c = floortiny(b + an / c, tiny)
        h *= d * c
    end
    return muladd(a, logt(x), -x) - loggamma(a) + logt(h)
end

# A function of its arguments alone, so the closures do not capture a loop variable.
@inline floortiny(v, tiny) = select(abs(v) < tiny, () -> tiny, () -> v)
