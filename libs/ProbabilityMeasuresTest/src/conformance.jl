"""
    default_ad_backends()

The automatic differentiation backends used by [`test_measure`](@ref). Pass Enzyme
through `ad_backends` when needed.
"""
function default_ad_backends()
    return (
        AutoForwardDiff(), AutoReverseDiff(), AutoZygote(), AutoMooncake(; config=nothing)
    )
end

# Keep the tuple length fixed so the compiler and Mooncake know the constructor's
# argument count.
"Rebuild `d` from the flattened parameters in `p`."
function _reconstruct(d, p)
    D = typeof(d)
    offsets, names = _paramoffsets(d), fieldnames(D)
    return constructorof(D)(
        ntuple(
            i -> _unflattenfield(d, names[i], getfield(d, i), p, offsets[i]),
            Val(fieldcount(D)),
        )...,
    )
end

# Flatten parameters into floating-point values that differentiation tools can change.
_paramvec(d) = reduce(vcat, _mapfields(_flattenfield, d))

_flatten(θ::Number) = [float(θ)]
_flatten(θ::AbstractArray) = vec(float.(θ))

_unflatten(::Number, p, offset::Int) = p[offset]
_unflatten(θ::AbstractArray, p, offset::Int) = reshape(p[_range(θ, offset)], size(θ))

_range(θ, offset::Int) = offset:(offset + length(θ) - 1)

"""
Names of the parameters that test sweeps must leave unchanged.

Name them rather than exclude them by type: `Binomial.n` fixes the support and the loop
lengths, while other integer parameters should still be swept.
"""
_structural(d) = ()

_isstructural(d, name::Symbol) = name in _structural(d)

function _mapfields(f, d)
    θs = params(d)
    return map((name, θ) -> f(d, name, θ), keys(θs), values(θs))
end

_flattenfield(d, name::Symbol, θ) = _isstructural(d, name) ? Union{}[] : _flatten(θ)
_lengthfield(d, name::Symbol, θ) = _isstructural(d, name) ? 0 : _paramlength(θ)

function _unflattenfield(d, name::Symbol, θ, p, offset::Int)
    return _isstructural(d, name) ? θ : _unflatten(θ, p, offset)
end

# Preserve structured parameters when flattening and rebuilding.
_flatten(θ::Diagonal) = _flatten(θ.diag)
_paramlength(θ::Diagonal) = length(θ.diag)
_unflatten(θ::Diagonal, p, offset::Int) = Diagonal(p[_range(θ.diag, offset)])

_flatten(θ::UniformScaling) = [float(θ.λ)]
_paramlength(::UniformScaling) = 1
_unflatten(::UniformScaling, p, offset::Int) = p[offset] * I

"Starting index of each parameter in `_paramvec(d)`."
function _paramoffsets(d)
    lengths = _mapfields(_lengthfield, d)
    # Zygote cannot handle the keyword call needed by `sum` here.
    return ntuple(Val(length(lengths))) do i
        offset = 1
        for j in 1:(i - 1)
            offset += lengths[j]
        end
        return offset
    end
end

_paramlength(::Number) = 1
_paramlength(θ::AbstractArray) = length(θ)

"Rebuild `d` with every parameter converted to `T`."
_withtype(d, ::Type{T}) where {T} = _reconstruct(d, map(T, _paramvec(d)))

# Convert evaluation points to `T` without losing their structure.
_aspoint(x::Number, ::Type{T}) where {T} = T(x)
_aspoint(x::AbstractArray, ::Type{T}) where {T} = T.(x)
_aspoint(x::UniformScaling, ::Type{T}) where {T} = T(x.λ) * I

# Use small rational approximations. Exact float conversion creates huge denominators
# that can overflow when a density squares them.
_asexact(x::Number, ::Type{I}) where {I<:Integer} = rationalize(I, x; tol=1//1000)
_asexact(x::AbstractArray, ::Type{I}) where {I<:Integer} = _asexact.(x, I)

_elscalar(d) = eltype(eltype(d))

# Gradient checks need a scalar result.
_scalarize(x::Number) = x
_scalarize(x::AbstractArray) = sum(x)

"""
    test_measure(d; kwargs...)

Run all standard checks for `d`.

# Keywords

  - `name::AbstractString`: the testset name. Defaults to the type name of `d`.
  - `xs`: evaluation points. Defaults to quantiles spanning the bulk and both tails.
  - `types`: the floating-point types swept for genericity.
  - `ad_backends`: see [`default_ad_backends`](@ref).
  - `reference_logpdf`: an optional `(d, x) -> Real` to check numerics against, for
    example a Distributions.jl equivalent.
  - `nsamples::Int`: Monte Carlo sample count for the moment checks.
  - `check_*::Bool`: force an individual block on or off.

# Defaults

Interface, invalid-input, numeric-type, compiler, and differentiation checks run for
every measure. Other checks run when the measure supports them:

  - normalization integrates continuous measures and sums discrete ones;
  - CDF checks require the optional CDF methods;
  - allocation, moment, and GPU checks require scalar samples;
  - mixed-type checks require more than one scalar parameter;
  - discrete samples skip the sample-derivative check;
  - Reactant checks run when its extension is loaded.

Checks skipped by these defaults belong in the measure's own test file.
"""
function test_measure(
    d;
    name::AbstractString=string(nameof(typeof(d))),
    xs=default_testpoints(d),
    types=(Float32, Float64, BigFloat),
    ad_backends=default_ad_backends(),
    reference_logpdf=nothing,
    nsamples::Int=200_000,
    check_interface::Bool=true,
    check_totality::Bool=true,
    check_genericity::Bool=true,
    check_inference::Bool=true,
    check_allocations::Bool=_can_check_allocations(d),
    check_normalization::Bool=_can_integrate(d) || _can_enumerate(d),
    check_cdf::Bool=_has_cdf(d),
    check_moments::Bool=_can_check_moments(d),
    check_ad::Bool=true,
    check_gpu::Bool=_can_gpu(d),
    check_reactant::Bool=_reactant_loaded(),
)
    @testset "$name" begin
        check_interface && @testset "interface" begin
            # `Interfaces.test` returns a boolean, so wrap it in `@test`.
            @test Interfaces.test(MeasureInterface, typeof(d), [d])
        end
        check_totality && @testset "invalid and extreme inputs" test_totality(d, xs)
        check_genericity && @testset "numeric types" test_genericity(d, xs, types)
        check_inference && @testset "type stability" test_inference(d, xs)
        check_allocations && @testset "allocations" test_allocations(d, xs)
        check_normalization && @testset "normalization" test_normalization(d)
        check_cdf && @testset "distribution function" test_cdf(d, xs)
        check_moments && @testset "moments" test_moments(d, nsamples)
        check_ad && @testset "automatic differentiation" test_ad(d, xs, ad_backends)
        check_gpu && @testset "GPU broadcast" test_gpu(d, xs)
        check_reactant && @testset "Reactant" test_reactant(d, xs)
        if reference_logpdf !== nothing
            @testset "reference numerics" begin
                for x in xs
                    @test logdensityof(d, x) ≈ reference_logpdf(d, x)
                end
            end
        end
    end
end

# Integration requires a continuous scalar measure with known endpoints.
function _can_integrate(d)
    d isa ContinuousMeasure || return false
    d isa UnivariateMeasure || return false
    return _bounded(support(d))
end

# Discrete normalization enumerates the support, so it needs a finite last outcome.
function _can_enumerate(d)
    d isa DiscreteMeasure || return false
    d isa UnivariateMeasure || return false
    _bounded(support(d)) || return false
    return isfinite(maximum(support(d)))
end

# Ignore Base's generic methods when checking for support bounds.
function _bounded(s)
    return _dispatches_on(minimum, (typeof(s),)) && _dispatches_on(maximum, (typeof(s),))
end

_has_scalar_params(d) = all(T -> T <: Number, fieldtypes(typeof(d)))

# Discrete samples change in steps, so their derivative is zero almost everywhere.
_is_reparameterized(d) = !(d isa DiscreteMeasure)

# `hasmethod` sees generic methods on `Any`. Make sure the selected method has a more
# specific first argument.
function _dispatches_on(f, argtypes::Tuple)
    D = Tuple{argtypes...}
    hasmethod(f, D) || return false
    sig = which(f, D).sig
    params = Base.unwrap_unionall(sig).parameters
    # The first entry is the function type; the second is its first argument.
    return length(params) >= 2 && params[2] !== Any
end

# Check for CDF support without drawing a sample.
_has_cdf(d) = _dispatches_on(cdf, (typeof(d), eltype(d)))

# Generic moment checks require scalar summaries.
function _can_check_moments(d)
    d isa UnivariateMeasure || return false
    D = (typeof(d),)
    return _dispatches_on(mean, D) && _dispatches_on(var, D) && _dispatches_on(std, D)
end

# Vector measures may need temporary storage and provide their own allocation tests.
_can_check_allocations(d) = d isa UnivariateMeasure

# The generic GPU test broadcasts over scalar points.
_can_gpu(d) = d isa UnivariateMeasure

"Evaluation points spanning the bulk and both tails of `d`."
function default_testpoints(d)
    ps = (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
    return [float(quantile(d, p)) for p in ps]
end

function test_totality(d, xs)
    # Models and GPU kernels may pass extreme values, so none of these may throw.
    for x in _extremepoints(d)
        @test (logdensityof(d, x); true)
    end
    for x in xs
        @test isfinite(logdensityof(d, x))
    end

    # Invalid parameters may return either `NaN` or `-Inf`.
    for bad in _invalids(d)
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, first(xs)))
    end
end

"Instances of `typeof(d)` with invalid parameters; empty if none are known."
_invalids(d) = ()

"Inputs used to check that `logdensityof` is total."
_extremepoints(d) = (Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64), 0.0)

function test_genericity(d, xs, types)
    for T in types
        dT = _withtype(d, T)
        x = _aspoint(first(xs), T)
        @test logdensityof(dT, x) isa T
        @test rand(Xoshiro(1), dT) isa eltype(dT)
        @test _elscalar(dT) === T
    end

    # Build mixed parameters field by field; a vector would give them one common type.
    mixed = _mixedparams(d)
    if mixed !== nothing
        # Containers can differ while still holding the same element type.
        @test length(unique(map(eltype, values(params(mixed))))) > 1
        @test logdensityof(mixed, _aspoint(first(xs), Float32)) isa Float64
    end

    # Exact parameters must not reduce the argument's precision.
    exact = _exactparams(d)
    if exact !== nothing
        x = first(xs)
        @test logdensityof(exact, _aspoint(x, Float32)) isa Float32
        vbig = logdensityof(exact, _aspoint(x, BigFloat))
        @test vbig isa BigFloat
        # Compare with parameters already converted to `BigFloat` to catch any hidden
        # `Float64` calculation.
        widened = logdensityof(_withtype(exact, BigFloat), _aspoint(x, BigFloat))
        @test abs(vbig - widened) < 1e-70

        test_exactness(exact, xs)
    end
end

"A test instance with exact parameters, using rationals when integers are not suitable."
_exactparams(d) = nothing

# Exact rational inputs expose calculations that assume floating-point parameters.
# `Rational{BigInt}` also reveals any hidden `Float64` intermediate.
function test_exactness(exact, xs)
    for I in (Int, BigInt)
        R = Rational{I}
        dR = _withtype(exact, R)
        for x in xs
            xR = _asexact(x, I)
            v = logdensityof(dR, xR)
            @test v isa float(R)
            widened = logdensityof(_withtype(exact, float(R)), _aspoint(xR, float(R)))
            @test isfinite(v) == isfinite(widened)
            isfinite(v) && @test v ≈ widened
        end
    end
end

# Use `Float32` for the first sweepable parameter and `Float64` for the rest.
function _mixedparams(d)
    D = typeof(d)
    θs = params(d)
    free = map(name -> !_isstructural(d, name), keys(θs))
    count(free) >= 2 || return nothing
    firstfree = findfirst(free)
    types = ntuple(i -> i == firstfree ? Float32 : Float64, Val(fieldcount(D)))
    # Fixed parameters keep their original type.
    converted = ntuple(Val(fieldcount(D))) do i
        θ = values(θs)[i]
        return free[i] ? _aspoint(θ, types[i]) : θ
    end
    return constructorof(D)(converted...)
end

function test_inference(d, xs)
    x = first(xs)
    @test (@inferred logdensityof(d, x)) isa Real
    @test (@inferred rand(Xoshiro(1), d)) isa eltype(d)
    JET.@test_opt target_modules = (ProbabilityMeasures,) logdensityof(d, x)
    JET.@test_call target_modules = (ProbabilityMeasures,) logdensityof(d, x)
end

function test_allocations(d, xs)
    # AllocCheck checks every call path rather than one warmed-up execution.
    @test isempty(check_allocs(logdensityof, (typeof(d), typeof(first(xs)))))
    @test isempty(check_allocs(rand, (Xoshiro, typeof(d))))
end

function test_normalization(d)
    lo, hi = _quadlimits(d)
    total, err = quadgk(x -> densityof(d, x), lo, hi; rtol=1e-10)
    @test total ≈ 1 atol = max(1e-8, 10err)
end

function test_normalization(d::DiscreteMeasure)
    s = support(d)
    @test sum(x -> densityof(d, float(x)), minimum(s):maximum(s)) ≈ 1
end

# QuadGK needs at least `Float64` limits to meet this tolerance.
function _quadlimits(d)
    s = support(d)
    return _widen(minimum(s)), _widen(maximum(s))
end

_widen(x) = convert(promote_type(typeof(float(x)), Float64), x)

function test_cdf(d, xs)
    for x in xs
        c = cdf(d, x)
        @test 0 <= c <= 1
        @test cdf(d, x) + ccdf(d, x) ≈ 1
        @test quantile(d, c) ≈ x rtol = 1e-6
        # Relative error is not useful when a log probability is near zero.
        @test logcdf(d, x) ≈ log(c) rtol = 1e-8 atol = 1e-12
        @test logccdf(d, x) ≈ log(ccdf(d, x)) rtol = 1e-8 atol = 1e-12
    end

    # A log-CDF should stay finite after the CDF underflows. Skip bounded measures when
    # rounding places the test point exactly on the lower endpoint.
    deep = float(quantile(d, 1e-300))
    if isfinite(deep) && deep > minimum(support(d))
        @test isfinite(logcdf(d, deep))
    end

    # Invalid probabilities must not make `quantile` throw.
    for p in (-0.001, 1.001, -Inf, Inf, NaN)
        @test (quantile(d, p); true)
    end

    # Check these types separately to catch hidden `Float64` calculations.
    for T in (Float32, Float64, BigFloat)
        dT = _withtype(d, T)
        xT = T(first(xs))
        @test cdf(dT, xT) isa T
        @test ccdf(dT, xT) isa T
        @test logcdf(dT, xT) isa T
        @test logccdf(dT, xT) isa T
        @test quantile(dT, T(1) / 4) isa T
    end

    # Only continuous CDFs can recover an arbitrary probability exactly.
    if d isa ContinuousMeasure
        setprecision(BigFloat, 256) do
            dbig = _withtype(d, BigFloat)
            p = big"0.25"
            @test abs(cdf(dbig, quantile(dbig, p)) - p) < 1e-60
        end
    end

    # For a continuous measure, the CDF is the integral of its density.
    if _can_integrate(d)
        lo = first(_quadlimits(d))
        x = _widen(quantile(d, 0.3))
        integral, _ = quadgk(t -> densityof(d, t), lo, x; rtol=1e-10)
        @test integral ≈ cdf(d, x) rtol = 1e-6
    end
end

function test_moments(d, nsamples)
    rng = Xoshiro(20250801)
    draws = rand(rng, d, nsamples)
    m, v = mean(draws), var(draws)
    # Allow five standard errors for the sampled mean.
    tol = 5 * std(d) / sqrt(nsamples)
    @test m ≈ mean(d) atol = tol
    @test v ≈ var(d) rtol = 20 / sqrt(nsamples)
    @test median(d) ≈ quantile(d, 0.5)
    @test std(d) ≈ sqrt(var(d))
end

function test_ad(d, xs, backends; check_reparameterization=_is_reparameterized(d))
    x = first(xs)
    p0 = _paramvec(d)
    # Use at least `Float64` for the finite-difference reference.
    p0_ref = convert.(promote_type(eltype(p0), Float64), p0)
    f = p -> logdensityof(_reconstruct(d, p), x)
    reference = FiniteDifferences.grad(central_fdm(5, 1), f, p0_ref)[1]

    # Keep the parameter type unchanged so both calls draw the same noise.
    draw = p -> _scalarize(rand(Xoshiro(7), _reconstruct(d, p)))
    draw_reference = if check_reparameterization
        FiniteDifferences.grad(central_fdm(5, 1), draw, p0)[1]
    else
        nothing
    end

    for backend in backends
        @testset "$(nameof(typeof(backend)))" begin
            g = DifferentiationInterface.gradient(f, backend, p0)
            @test g ≈ reference rtol = 1e-5 atol = 1e-8
            if check_reparameterization
                @testset "sample derivative" test_reparameterization(
                    draw, p0, draw_reference, backend
                )
            end
        end
    end
end

"Check the derivative of `draw` with respect to its parameters."
function test_reparameterization(draw, p0, reference, backend)
    g = DifferentiationInterface.gradient(draw, backend, p0)
    @test g ≈ reference rtol = 1e-5 atol = 1e-8
    # A zero gradient means the sample does not depend on its parameters.
    @test any(!iszero, g)
end

function test_gpu(d, xs)
    # JLArray checks GPU broadcasting rules without physical GPU hardware.
    d32 = _withtype(d, Float32)
    x32 = Float32.(xs)
    expected = logdensityof.(d32, x32)

    # Measures with scalar parameters must fit directly in the GPU operation.
    _has_scalar_params(d) && @test isbits(d32)

    # Disallow scalar indexing for this check, then restore the caller's setting.
    saved = get(task_local_storage(), :ScalarIndexing, nothing)
    task_local_storage(:ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    try
        got = Array(logdensityof.(d32, JLArray(x32)))
        @test got ≈ expected
        @test eltype(got) === Float32
    finally
        if saved === nothing
            delete!(task_local_storage(), :ScalarIndexing)
        else
            task_local_storage(:ScalarIndexing, saved)
        end
    end
end

"""
    test_reactant(d, xs)

Check that Reactant can compile `d` with wrapped parameters and data.

The implementation lives in `ext/ProbabilityMeasuresTestReactantExt.jl` and runs when
that extension is loaded. Pass `check_reactant=true` to require it.
"""
function test_reactant(::Any, ::Any)
    return error("test_reactant needs Reactant. Run `using Reactant` first.")
end

"Whether the Reactant extension of this package has loaded."
function _reactant_loaded()
    return Base.get_extension(@__MODULE__, :ProbabilityMeasuresTestReactantExt) !== nothing
end
