"""
    logt(x)

Total `log`: returns `NaN` where `log` would throw a `DomainError`.

`logdensityof` must never throw (invariant 2 of
[`AbstractProbabilityMeasure`](@ref)) because a throw inside a GPU kernel is
undefined behaviour and a PPL will hand these functions invalid parameters during
line search, warmup, and rejected proposals. `log(0)` is already `-Inf` and does not
throw, so only the negative branch needs handling.
"""
@inline logt(x::Real) = x < zero(x) ? oftype(float(x), NaN) : log(x)

"""
    basefloat(T) -> Type{<:AbstractFloat}

The plain floating-point type underlying `T`, with any AD tracking removed.

Used by [`noisetype`](@ref) to decide the type of the *underlying randomness* in a
reparameterized draw. Sampling `randn` in the tracked type would be wrong (and
usually unsupported); the tracking must enter through the parameters instead, which
is exactly what makes the draw differentiable.

AD packages extend this via package extensions -- see
`ext/ProbabilityMeasuresForwardDiffExt.jl`.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{T}) where {T<:Real} = float(T)
basefloat(::Type{Bool}) = Float64
basefloat(::Type{<:Irrational}) = Float64
