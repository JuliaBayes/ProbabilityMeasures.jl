module ProbabilityMeasuresReverseDiffExt

using ProbabilityMeasures: ProbabilityMeasures
using ReverseDiff: TrackedReal, value

# `float(TrackedReal)` is still wrapped, but random noise needs the plain type.
function ProbabilityMeasures.basefloat(::Type{TrackedReal{V,D,O}}) where {V,D,O}
    return ProbabilityMeasures.basefloat(V)
end

# Keep a rejection sampler's accept step off the tape.
function ProbabilityMeasures.basevalue(x::TrackedReal)
    return ProbabilityMeasures.basevalue(value(x))
end

end
