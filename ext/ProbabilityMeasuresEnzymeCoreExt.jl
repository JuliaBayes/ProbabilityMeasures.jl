module ProbabilityMeasuresEnzymeCoreExt

using EnzymeCore.EnzymeRules: EnzymeRules
using ProbabilityMeasures: ProbabilityMeasures

# Enzyme can differentiate the arithmetic directly. These helpers have no derivative.

EnzymeRules.inactive(::typeof(ProbabilityMeasures.checkparams), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.support), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.insupport), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.noisetype), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.basefloat), args...) = nothing

# A rejection sampler's accept step is a choice, not a value the draw depends on.
EnzymeRules.inactive(::typeof(ProbabilityMeasures.basevalue), args...) = nothing

# Supports contain no differentiable data.
EnzymeRules.inactive_type(::Type{<:ProbabilityMeasures.Support}) = true

end
