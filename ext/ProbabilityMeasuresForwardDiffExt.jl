module ProbabilityMeasuresForwardDiffExt

using ForwardDiff: Dual
using ProbabilityMeasures: ProbabilityMeasures

# Draw noise in the plain float type. Dual parameters still affect the returned sample.
function ProbabilityMeasures.basefloat(::Type{<:Dual{T,V,N}}) where {T,V,N}
    return ProbabilityMeasures.basefloat(V)
end

end
