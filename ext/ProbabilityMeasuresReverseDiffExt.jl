module ProbabilityMeasuresReverseDiffExt

using ProbabilityMeasures: ProbabilityMeasures
using ReverseDiff: TrackedReal

# `float(TrackedReal)` is still wrapped, but random noise needs the plain type.
function ProbabilityMeasures.basefloat(::Type{TrackedReal{V,D,O}}) where {V,D,O}
    return ProbabilityMeasures.basefloat(V)
end

end
