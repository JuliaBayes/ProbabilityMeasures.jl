module ProbabilityMeasuresReactantExt

using ProbabilityMeasures: ProbabilityMeasures
using Reactant: Reactant, TracedRNumber
using SpecialFunctions: SpecialFunctions

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

# Reactant exposes the underlying operation but not these SpecialFunctions methods.
# Remove them once Reactant provides the bindings.
function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(one(x) - x)
end

function SpecialFunctions.erfinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(x)
end

end
