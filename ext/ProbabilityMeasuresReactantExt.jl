module ProbabilityMeasuresReactantExt

using ProbabilityMeasures: ProbabilityMeasures
using ProbabilityMeasures: BETAINC_TERMS, BETAQUANTILE_STEPS, GAMMAINC_TERMS
using ProbabilityMeasures: GAMMAQUANTILE_STEPS, basefloat, floortiny, log1pt, logt, select
using ProbabilityMeasures: betaquantile_start, betaquantile_step, logbetat, logistic
using ProbabilityMeasures: gammaquantile_start, gammaquantile_step
using Reactant: Reactant, TracedRNumber, @trace
using SpecialFunctions: SpecialFunctions, loggamma

# Reactant's random functions already wrap their result, so request the plain element
# type instead of wrapping it twice.
function ProbabilityMeasures.basefloat(::Type{TracedRNumber{T}}) where {T}
    return ProbabilityMeasures.basefloat(T)
end

# Base cannot find the floating-point form of this wrapped type. Remove this method
# once Reactant provides it.
Base.float(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{float(T)}

# Reactant evaluates both choices, so each must be safe to call.
@inline function ProbabilityMeasures.select(cond::TracedRNumber{Bool}, iftrue, iffalse)
    return ifelse(cond, iftrue(), iffalse())
end

# Comparisons on traced numbers cannot stop a Julia loop, so loops run a fixed length.
ProbabilityMeasures.wrappedconditions(::Type{<:TracedRNumber}) = true

#=
  The fixed-length loops in `gammainc.jl` and `gamma.jl` are plain `for` loops, which
  tracing would unroll into thousands of operations and take minutes to compile. These
  methods repeat them under `@trace`, which keeps each loop a single operation. The
  loop bodies must not mention the static parameter `T`, which `@trace` would shadow.
=#
function ProbabilityMeasures.gammap_series_fixed(a::T, x::T) where {T<:TracedRNumber}
    term = one(T)
    total = one(T)
    @trace for n in 1:GAMMAINC_TERMS
        term = term * (x / (a + n))
        total = total + term
    end
    return muladd(a, logt(x), -x) - loggamma(a + one(T)) + logt(total)
end

function ProbabilityMeasures.gammaq_cf_fixed(a::T, x::T) where {T<:TracedRNumber}
    F = basefloat(T)
    tiny = convert(T, sqrt(floatmin(F)))
    b = x + one(T) - a
    c = convert(T, inv(sqrt(floatmin(F))))
    # `@trace` needs loop-carried variables to be distinct objects.
    d = inv(b)
    h = inv(b)
    two = 2 * one(T)
    @trace for i in 1:GAMMAINC_TERMS
        an = -i * (i - a)
        b = b + two
        d = inv(floortiny(an * d + b, tiny))
        c = floortiny(b + an / c, tiny)
        h = h * (d * c)
    end
    return muladd(a, logt(x), -x) - loggamma(a) + logt(h)
end

function ProbabilityMeasures.gammalogquantile_fixed(a::T, p::T) where {T<:TracedRNumber}
    lower = p <= one(T) / 2
    target = select(lower, () -> logt(p), () -> log1pt(-p))
    lg = loggamma(a)
    u = gammaquantile_start(a, p)
    @trace for k in 1:GAMMAQUANTILE_STEPS
        u = u - gammaquantile_step(a, lg, u, exp(u), lower, target)
    end
    return u
end

function ProbabilityMeasures.betainc_cf_fixed(a::T, b::T, x::T) where {T<:TracedRNumber}
    F = basefloat(T)
    tiny = convert(T, sqrt(floatmin(F)))
    unit = one(T)
    qab, qap, qam = a + b, a + one(T), a - one(T)
    c = one(T)
    d = inv(floortiny(one(T) - qab * x / qap, tiny))
    h = inv(floortiny(one(T) - qab * x / qap, tiny))
    @trace for m in 1:BETAINC_TERMS
        m2 = 2m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = inv(floortiny(unit + aa * d, tiny))
        c = floortiny(unit + aa / c, tiny)
        h = h * (d * c)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = inv(floortiny(unit + aa * d, tiny))
        c = floortiny(unit + aa / c, tiny)
        h = h * (d * c)
    end
    return h
end

function ProbabilityMeasures.betaquantile_fixed(a::T, b::T, p::T) where {T<:TracedRNumber}
    lower = p <= one(T) / 2
    target = select(lower, () -> logt(p), () -> log1pt(-p))
    lb = logbetat(a, b)
    v = betaquantile_start(a, b, p)
    @trace for k in 1:BETAQUANTILE_STEPS
        v = v - betaquantile_step(a, b, lb, v, lower, target)
    end
    return logistic(v)
end

# Reactant exposes the underlying operation but not these SpecialFunctions methods.
# Remove them once Reactant provides the bindings.
function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(one(x) - x)
end

function SpecialFunctions.erfinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(x)
end

end
