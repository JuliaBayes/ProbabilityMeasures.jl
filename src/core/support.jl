"""
    Support

Supertype for descriptions of the set a measure puts its mass on.

Supports are deliberately *singletons wherever possible*. A singleton carries no
numeric payload, so it cannot pin a precision or leak a `Float64` into an otherwise
`Float32` computation, and it costs nothing to pass into a kernel.
"""
abstract type Support end

"The whole real line, ``(-\\infty, \\infty)``."
struct RealLine <: Support end

"The positive reals, ``(0, \\infty)``."
struct PositiveReals <: Support end

"The unit interval ``[0, 1]``."
struct UnitInterval <: Support end

"""
    RealInterval(lower, upper)

A bounded interval ``[lower, upper]``. Unlike the singleton supports this carries
values, so construct it with the measure's own parameter types to stay generic.
"""
struct RealInterval{L<:Real,U<:Real} <: Support
    lower::L
    upper::U
end

"""
    support(d) -> Support

The set on which `d` places its mass.
"""
function support end

"""
    insupport(d, x) -> Bool
    insupport(s::Support, x) -> Bool

Whether `x` lies in the support.

This is a *predicate*, not a precondition: [`logdensityof`](@ref) is total and
returns `-Inf` outside the support on its own, so densities do not need to call
this. It exists for samplers, transforms, and validation.
"""
insupport(d::AbstractProbabilityMeasure, x) = insupport(support(d), x)

insupport(::RealLine, x::Real) = isfinite(x)
insupport(::PositiveReals, x::Real) = isfinite(x) && x > zero(x)
insupport(::UnitInterval, x::Real) = zero(x) <= x <= one(x)
insupport(s::RealInterval, x::Real) = s.lower <= x <= s.upper

Base.minimum(::RealLine) = -Inf
Base.maximum(::RealLine) = Inf
Base.minimum(::PositiveReals) = 0.0
Base.maximum(::PositiveReals) = Inf
Base.minimum(::UnitInterval) = 0.0
Base.maximum(::UnitInterval) = 1.0
Base.minimum(s::RealInterval) = s.lower
Base.maximum(s::RealInterval) = s.upper
