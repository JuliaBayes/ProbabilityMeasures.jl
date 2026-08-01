module ProbabilityMeasuresForwardDiffExt

using ForwardDiff: Dual
using ProbabilityMeasures: ProbabilityMeasures

# Without this, `rand(rng, Normal(dual_μ, dual_σ))` would try to draw the underlying
# standard normal *in the dual type*, which is both meaningless and unsupported. The
# randomness belongs in the plain float type; the duals enter through `μ + σ * z`,
# which is exactly the reparameterization that makes the draw differentiable.
function ProbabilityMeasures.basefloat(::Type{<:Dual{T,V,N}}) where {T,V,N}
    return ProbabilityMeasures.basefloat(V)
end

end
