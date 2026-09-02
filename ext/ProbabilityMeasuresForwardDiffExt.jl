module ProbabilityMeasuresForwardDiffExt

using ForwardDiff: Dual, value
using ProbabilityMeasures: ProbabilityMeasures

# Draw noise in the plain float type. Dual parameters still affect the returned sample.
function ProbabilityMeasures.basefloat(::Type{<:Dual{T,V,N}}) where {T,V,N}
    return ProbabilityMeasures.basefloat(V)
end

# Keep a rejection sampler's accept step off the derivative.
function ProbabilityMeasures.basevalue(x::Dual)
    return ProbabilityMeasures.basevalue(value(x))
end

end
