#=
  The regularized incomplete gamma integrals, in log space. Returning logarithms keeps
  a tail readable where the probability itself underflows, and every step stays in the
  type of the arguments, so `BigFloat` keeps its precision.
=#

#=
  Both expansions converge geometrically once `x` is on their own side of `a + 1`. At
  the crossover they need about `8.5√a` terms in `Float64`, and more at higher
  precision, so this bound covers `Float64` shapes up to roughly `10^6`. Past it the
  remaining terms are dropped and the result loses accuracy.
=#
const GAMMAINC_MAXITER = 10_000

"""
    loggammap(a, x)

`log(P(a, x))`, where ``P`` is the lower incomplete gamma integral divided by
``\\Gamma(a)``:

```math
P(a, x) = \\frac{1}{\\Gamma(a)} \\int_0^x t^{a-1} e^{-t} \\, \\mathrm{d}t.
```

`a` must be positive and `x` non-negative. Below `x = a + 1` this sums a series whose
terms are all positive; above it, it takes the complement of [`loggammaq`](@ref). Both
loop until the terms stop changing the result, which rules out tracing and device-side
evaluation.

The prefactor is a difference of terms of order ``a \\log a``, so relative accuracy
holds to the rounding error of the argument type for shapes up to about `1000` and then
falls off roughly in proportion to the shape.

See also [`loggammaq`](@ref).
"""
@inline function loggammap(a::T, x::T) where {T<:Number}
    return x < a + one(T) ? gammap_series(a, x) : log1mexpt(gammaq_cf(a, x))
end

"""
    loggammaq(a, x)

`log(Q(a, x))`, where ``Q(a, x) = 1 - P(a, x)`` is the upper incomplete gamma integral
divided by ``\\Gamma(a)``.

Above `x = a + 1` this evaluates a continued fraction; below it, it takes the
complement of [`loggammap`](@ref).
"""
@inline function loggammaq(a::T, x::T) where {T<:Number}
    return x < a + one(T) ? log1mexpt(gammap_series(a, x)) : gammaq_cf(a, x)
end

#=
  `P(a, x) = x^a e^{-x} / Γ(a+1) · Σₙ xⁿ / ((a+1)⋯(a+n))`. Every term is positive, so
  the sum loses nothing to cancellation.
=#
function gammap_series(a::T, x::T) where {T<:Number}
    tol = eps(basefloat(T))
    term = one(T)
    total = one(T)
    n = 0
    while (n < GAMMAINC_MAXITER) & (abs(term) > tol * abs(total))
        n += 1
        term *= x / (a + n)
        total += term
    end
    return muladd(a, logt(x), -x) - loggamma(a + one(T)) + logt(total)
end

#=
  `Q(a, x) = x^a e^{-x} / Γ(a) · CF`, with the continued fraction

      CF = 1/(b₀ - a₁/(b₁ - a₂/(b₂ - ⋯))),   b₀ = x + 1 - a, bᵢ = b₀ + 2i, aᵢ = i(i - a)

  evaluated by Lentz's method, which builds the fraction from the front and so needs no
  guess at where to truncate it. `tiny` replaces a denominator that rounds to zero,
  which is what lets the recurrence step past a vanishing partial numerator.
=#
function gammaq_cf(a::T, x::T) where {T<:Number}
    tol = eps(basefloat(T))
    tiny = convert(T, floatmin(basefloat(T))) / tol
    b = x + one(T) - a
    c = inv(tiny)
    d = inv(b)
    h = d
    for i in 1:GAMMAINC_MAXITER
        an = -i * (i - a)
        b += 2 * one(T)
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        delta = d * c
        h *= delta
        abs(delta - one(T)) <= tol && break
    end
    return muladd(a, logt(x), -x) - loggamma(a) + logt(h)
end
