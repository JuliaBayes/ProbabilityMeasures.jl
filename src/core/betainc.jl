#=
  The regularized incomplete beta function and its inverse. SpecialFunctions exports
  `beta_inc` and `beta_inc_inv`, but only for `Float16`, `Float32` and `Float64`. These
  work for any `AbstractFloat`, which is what the `BigFloat` checks in the conformance
  suite need.
=#

"""
    betacf(a, b, x)

The continued fraction of Abramowitz & Stegun 26.5.8, evaluated by Lentz's method.

It converges quickly only for ``x < (a+1)/(a+b+2)``. [`betainc`](@ref) reaches larger
arguments through the symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)``.
"""
function betacf(a::T, b::T, x::T) where {T<:AbstractFloat}
    # Lentz's method stalls on an exact zero, so hold denominators away from it.
    tiny = floatmin(T)
    qab, qap, qam = a + b, a + one(T), a - one(T)
    c = one(T)
    d = one(T) - qab * x / qap
    abs(d) < tiny && (d = tiny)
    d = inv(d)
    h = d
    for m in 1:(2 * precision(T) + 100)
        m2 = 2m
        even = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = one(T) + even * d
        abs(d) < tiny && (d = tiny)
        c = one(T) + even / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        h *= d * c
        odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = one(T) + odd * d
        abs(d) < tiny && (d = tiny)
        c = one(T) + odd / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        step = d * c
        h *= step
        abs(step - one(T)) <= eps(T) && break
    end
    return h
end

"""
    betainc(a, b, x, xc) -> (p, q)
    betainc(a, b, x) -> (p, q)

The regularized incomplete beta ``I_x(a, b)`` and its complement ``1 - I_x(a, b)``.

`xc` is ``1 - x``. Passing it separately keeps the precision of an argument close to
one, and both tails come back computed directly rather than subtracted from one.
"""
function betainc(a::T, b::T, x::T, xc::T) where {T<:AbstractFloat}
    x <= zero(T) && return (zero(T), one(T))
    xc <= zero(T) && return (one(T), zero(T))
    lead = a * log(x) + b * log(xc) - logbeta(a, b)
    if x * (a + b + 2 * one(T)) < a + one(T)
        p = exp(lead) / a * betacf(a, b, x)
        return (p, one(T) - p)
    end
    q = exp(lead) / b * betacf(b, a, xc)
    return (one(T) - q, q)
end

betainc(a::T, b::T, x::T) where {T<:AbstractFloat} = betainc(a, b, x, one(T) - x)

# `I_x(a, b)` as a function of `u = log(x)`, which is where the inverse iterates.
@inline function betainclog(a::T, b::T, u::T) where {T<:AbstractFloat}
    return betainc(a, b, exp(u), -expm1(u))[1]
end

"""
    betaincinv(a, b, p, q) -> (x, xc)

The `x` with ``I_x(a, b) = p``, together with ``1 - x``.

`q` is ``1 - p``. The smaller of the two probabilities selects which side to solve for,
so the returned pair is accurate at both ends of the unit interval.
"""
function betaincinv(a::T, b::T, p::T, q::T) where {T<:AbstractFloat}
    p <= zero(T) && return (zero(T), one(T))
    q <= zero(T) && return (one(T), zero(T))
    p <= q && return betaincinvsmall(a, b, p)
    xc, x = betaincinvsmall(b, a, q)
    return (x, xc)
end

"""
    betaincinvsmall(a, b, p) -> (x, 1 - x)

Solve ``I_x(a, b) = p`` for ``p \\le 1/2`` by Newton's method in ``u = \\log x``.

Working in the logarithm keeps a small root accurate and returns both `x` and its
complement from `exp` and `expm1`. Each step falls back to bisecting a bracket that the
iteration maintains, so the root is always approached from a known interval.
"""
function betaincinvsmall(a::T, b::T, p::T) where {T<:AbstractFloat}
    lbeta = logbeta(a, b)
    # `I_x(a, b) ≈ x^a / (a B(a, b))` for small `x` gives the starting point.
    u = min((log(p) + log(a) + lbeta) / a, -eps(T))
    ulo, uhi = u, u
    if betainclog(a, b, u) > p
        step = one(T)
        while betainclog(a, b, ulo) > p
            uhi = ulo
            ulo -= step
            step += step
        end
    else
        while betainclog(a, b, uhi) < p
            ulo = uhi
            uhi /= 2
        end
    end

    for _ in 1:(2 * precision(T) + 100)
        x, xc = exp(u), -expm1(u)
        f = betainc(a, b, x, xc)[1] - p
        f < zero(T) ? (ulo = u) : (uhi = u)
        # The derivative of `I` in `u` is `x` times its derivative in `x`.
        slope = exp(a * u + (b - one(T)) * log(xc) - lbeta)
        unew = u - f / slope
        (isfinite(unew) && (ulo <= unew <= uhi)) || (unew = (ulo + uhi) / 2)
        # A few last bits are beyond the accuracy of `betainc` itself.
        done = abs(unew - u) <= 8 * eps(T) * abs(u)
        u = unew
        done && break
    end
    return (exp(u), -expm1(u))
end

"""
    logbetainc(a, b, x, xc)

`log(`[`betainc`](@ref)`(a, b, x, xc)[1])`, which stays finite where the tail itself
underflows.

Only the continued fraction's own branch gains anything. Above its crossover the result
is one minus a small number, whose logarithm is near zero and loses nothing to the
ordinary route.
"""
function logbetainc(a::T, b::T, x::T, xc::T) where {T<:AbstractFloat}
    x <= zero(T) && return convert(T, -Inf)
    xc <= zero(T) && return zero(T)
    if x * (a + b + 2 * one(T)) < a + one(T)
        lead = a * log(x) + b * log(xc) - logbeta(a, b)
        return lead - log(a) + log(betacf(a, b, x))
    end
    return log(betainc(a, b, x, xc)[1])
end
