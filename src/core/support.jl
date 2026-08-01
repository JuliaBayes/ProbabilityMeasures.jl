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

# `PositiveReals`, `UnitInterval` and a bounded `RealInterval` all belong here, and
# will arrive with the first measure that needs one (Gamma, Beta, Uniform). Defining
# them now would be guessing at their shape before anything constrains it.

"""
    support(d) -> Support

The set on which `d` places its mass.
"""
function support end

"""
    insupport(d, x) -> Bool
    insupport(s::Support, x) -> Bool

Whether `x` lies in the support.

This is a *predicate*, not a precondition: `logdensityof` is total and
returns `-Inf` outside the support on its own, so densities do not need to call
this. It exists for samplers, transforms, and validation.
"""
insupport(d::AbstractProbabilityMeasure, x) = insupport(support(d), x)

insupport(::RealLine, x::Real) = isfinite(x)

Base.minimum(::RealLine) = -Inf
Base.maximum(::RealLine) = Inf
