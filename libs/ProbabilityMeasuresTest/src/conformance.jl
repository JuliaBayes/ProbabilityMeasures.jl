#=
  The conformance suite. `test_measure` is the single entry point: every measure in
  the package must pass it, and it is the mechanism by which the type-genericity,
  allocation, AD and GPU claims in the README are actually enforced rather than
  merely asserted.
=#

"""
    default_ad_backends()

The AD backends exercised by [`test_measure`](@ref) by default.

Enzyme is deliberately absent: it is a heavy dependency whose Windows support is
uneven, and a conformance suite that cannot run everywhere is not much of a
conformance suite. Pass it explicitly via `ad_backends` to include it.
"""
function default_ad_backends()
    return (
        AutoForwardDiff(), AutoReverseDiff(), AutoZygote(), AutoMooncake(; config=nothing)
    )
end

"Rebuild `d` with parameters taken from the vector `p`."
_reconstruct(d, p) = constructorof(typeof(d))(p...)

#=
  The parameters of `d` as a flat vector, promoted to a common *floating-point*
  type. The `float` matters: a measure may legitimately carry integer parameters,
  but you cannot perturb an `Int` by a finite-difference step, and AD cannot track
  one either.
=#
_paramvec(d) = collect(promote(map(float, values(params(d)))...))

"Rebuild `d` with every parameter converted to `T`."
_withtype(d, ::Type{T}) where {T} = _reconstruct(d, map(T, _paramvec(d)))

"""
    test_measure(d; kwargs...)

Run the full conformance suite against the measure `d`.

Each block below corresponds to a property this package claims to guarantee. A new
measure is "done" when this passes.

# Keyword arguments

  - `xs`: evaluation points. Defaults to a spread of quantiles plus the mean.
  - `types`: the floating-point types swept for genericity.
  - `ad_backends`: see [`default_ad_backends`](@ref).
  - `reference_logpdf`: an optional `(d, x) -> Real` to check numerics against, e.g.
    a Distributions.jl equivalent.
  - `nsamples`: Monte Carlo sample count for the moment checks.
  - The `check_*` flags switch off individual blocks for measures that cannot
    support them.
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
    check_allocations::Bool=true,
    check_normalization::Bool=true,
    check_cdf::Bool=true,
    check_moments::Bool=true,
    check_ad::Bool=true,
    check_gpu::Bool=true,
)
    @testset "$name" begin
        check_interface && @testset "interface" begin
            # `Interfaces.test` prints a per-component report and returns a Bool;
            # without the `@test` the testset records nothing.
            @test Interfaces.test(MeasureInterface, typeof(d), [d])
        end
        check_totality && @testset "totality" test_totality(d, xs)
        check_genericity && @testset "type genericity" test_genericity(d, xs, types)
        check_inference && @testset "type stability" test_inference(d, xs)
        check_allocations && @testset "allocations" test_allocations(d, xs)
        check_normalization && @testset "normalization" test_normalization(d)
        check_cdf && @testset "distribution function" test_cdf(d, xs)
        check_moments && @testset "moments" test_moments(d, nsamples)
        check_ad && @testset "automatic differentiation" test_ad(d, xs, ad_backends)
        check_gpu && @testset "GPU broadcast" test_gpu(d, xs)
        if reference_logpdf !== nothing
            @testset "reference numerics" begin
                for x in xs
                    @test logdensityof(d, x) ≈ reference_logpdf(d, x)
                end
            end
        end
    end
end

"Evaluation points spanning the bulk and both tails of `d`."
function default_testpoints(d)
    ps = (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
    return [float(quantile(d, p)) for p in ps]
end

# --- Invariant 2: logdensityof is total ------------------------------------------

function test_totality(d, xs)
    # A throw here is undefined behaviour inside a GPU kernel, and a PPL will hand
    # these values in from a bad proposal or an overshooting line search.
    for x in (Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64), 0.0)
        @test (logdensityof(d, x); true)
    end
    for x in xs
        @test isfinite(logdensityof(d, x))
    end

    # Invalid parameters produce a non-finite value rather than an error, and
    # construction itself never complains.
    #
    # Deliberately *not* `isnan`: which non-finite value you get is not part of the
    # contract. `Normal(Inf, 1.0)` is invalid but has a log-density of -Inf, and
    # pinning the suite to NaN would push callers toward `isnan` as a validity
    # sentinel, which silently accepts exactly that case.
    for bad in _invalids(d)
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, first(xs)))
    end
end

"Instances of `typeof(d)` with invalid parameters; empty if none are known."
_invalids(d) = ()

# --- Invariant 1: type genericity -------------------------------------------------

function test_genericity(d, xs, types)
    for T in types
        dT = _withtype(d, T)
        x = T(first(xs))
        @test logdensityof(dT, x) isa T
        @test rand(Xoshiro(1), dT) isa T
        @test eltype(dT) === T
    end

    # Mixed parameter types must neither error nor widen past the true promotion:
    # one Float32 parameter alongside Float64 ones promotes to Float64, and no
    # further.
    #
    # A *tuple*, not a vector. `[Float32(a), Float64(b)]` is a `Vector{Float64}` --
    # the literal promotes and converts the Float32 straight back, so the measure
    # comes out homogeneous and the check passes without ever testing anything.
    p = _paramvec(d)
    if length(p) >= 2
        mixed = _reconstruct(d, (Float32(p[1]), Float64.(p[2:end])...))
        # Assert the measure really is mixed before drawing any conclusion from it.
        @test length(unique(fieldtypes(typeof(mixed)))) > 1
        @test logdensityof(mixed, Float32(first(xs))) isa Float64
    end

    # Exact (integer) parameters must not cap the precision of the result -- it has
    # to follow the argument.
    exact = _exactparams(d)
    if exact !== nothing
        x = first(xs)
        @test logdensityof(exact, Float32(x)) isa Float32
        vbig = logdensityof(exact, big(float(x)))
        @test vbig isa BigFloat
        # The same measure with the parameters already widened. If any Irrational
        # constant or `log` were evaluated at Float64 along the way, these would
        # agree only to ~1e-16 instead of to full BigFloat precision.
        @test abs(vbig - logdensityof(_withtype(exact, BigFloat), big(float(x)))) < 1e-70
    end
end

"An instance of `typeof(d)` with exact (integer) parameters, or `nothing`."
_exactparams(d) = nothing

# --- Type stability and allocations ------------------------------------------------

function test_inference(d, xs)
    x = first(xs)
    @test (@inferred logdensityof(d, x)) isa Real
    @test (@inferred rand(Xoshiro(1), d)) isa eltype(d)
    JET.@test_opt target_modules = (ProbabilityMeasures,) logdensityof(d, x)
    JET.@test_call target_modules = (ProbabilityMeasures,) logdensityof(d, x)
end

function test_allocations(d, xs)
    # AllocCheck proves this statically over the whole call graph, which is far
    # stronger than timing `@allocated` and hoping the benchmark was warm.
    @test isempty(check_allocs(logdensityof, (typeof(d), typeof(first(xs)))))
    @test isempty(check_allocs(rand, (Xoshiro, typeof(d))))
end

# --- Correctness -------------------------------------------------------------------

function test_normalization(d)
    s = support(d)
    total, err = quadgk(x -> densityof(d, x), minimum(s), maximum(s); rtol=1e-10)
    @test total ≈ 1 atol = max(1e-8, 10err)
end

function test_cdf(d, xs)
    for x in xs
        c = cdf(d, x)
        @test 0 <= c <= 1
        @test cdf(d, x) + ccdf(d, x) ≈ 1
        @test quantile(d, c) ≈ x rtol = 1e-6
        # `atol` as well as `rtol`: in the upper tail `log(c)` is a tiny negative
        # number and a purely relative comparison is meaningless there.
        @test logcdf(d, x) ≈ log(c) rtol = 1e-8 atol = 1e-12
        @test logccdf(d, x) ≈ log(ccdf(d, x)) rtol = 1e-8 atol = 1e-12
    end

    # The whole point of `logcdf` over `log(cdf(...))`: `cdf` underflows to zero far
    # out in the tail, where the log-scale value is still perfectly finite.
    deep = float(quantile(d, 1e-300))
    if isfinite(deep)
        @test isfinite(logcdf(d, deep))
    end

    # The distribution function has to be as type-generic as the density is. Checking
    # only `logdensityof` let a `Float64`-collapsing `quantile` through: `-sqrt2 * x`
    # parses as `(-sqrt2) * x`, and negating an Irrational materializes it at
    # Float64 before it ever sees the argument.
    for T in (Float32, Float64, BigFloat)
        dT = _withtype(d, T)
        xT = T(first(xs))
        @test cdf(dT, xT) isa T
        @test ccdf(dT, xT) isa T
        @test logcdf(dT, xT) isa T
        @test logccdf(dT, xT) isa T
        @test quantile(dT, T(1) / 4) isa T
    end

    # ...and as precise. A Float64 intermediate anywhere in the chain caps this at
    # ~1e-17 instead of full BigFloat precision.
    setprecision(BigFloat, 256) do
        dbig = _withtype(d, BigFloat)
        p = big"0.25"
        @test abs(cdf(dbig, quantile(dbig, p)) - p) < 1e-60
    end

    # cdf must be the integral of the density.
    lo = minimum(support(d))
    x = float(quantile(d, 0.3))
    integral, _ = quadgk(t -> densityof(d, t), lo, x; rtol=1e-10)
    @test integral ≈ cdf(d, x) rtol = 1e-6
end

function test_moments(d, nsamples)
    rng = Xoshiro(20250801)
    draws = rand(rng, d, nsamples)
    m, v = mean(draws), var(draws)
    # Monte Carlo error on the mean is std/sqrt(n); allow five of them.
    tol = 5 * std(d) / sqrt(nsamples)
    @test m ≈ mean(d) atol = tol
    @test v ≈ var(d) rtol = 20 / sqrt(nsamples)
    @test median(d) ≈ quantile(d, 0.5)
    @test std(d) ≈ sqrt(var(d))
end

# --- Automatic differentiation -------------------------------------------------------

function test_ad(d, xs, backends)
    x = first(xs)
    p0 = _paramvec(d)
    f = p -> logdensityof(_reconstruct(d, p), x)
    reference = FiniteDifferences.grad(central_fdm(5, 1), f, p0)[1]

    for backend in backends
        @testset "$(nameof(typeof(backend)))" begin
            g = DifferentiationInterface.gradient(f, backend, p0)
            @test g ≈ reference rtol = 1e-5 atol = 1e-8
        end
    end

    # Sampling is written in reparameterized form, so the pathwise derivative must
    # exist and be exact -- this is what a VI backend in the PPL relies on.
    @testset "reparameterized rand" test_reparameterization(d)
end

"Check that `rand` is differentiable with respect to the parameters."
function test_reparameterization(d)
    p0 = _paramvec(d)
    draw = p -> rand(Xoshiro(7), _reconstruct(d, p))
    g = ForwardDiff.gradient(draw, p0)
    reference = FiniteDifferences.grad(central_fdm(5, 1), draw, p0)[1]
    @test g ≈ reference rtol = 1e-5 atol = 1e-8
    # A zero gradient would mean the draw does not actually depend on the
    # parameters, i.e. the reparameterization is broken.
    @test any(!iszero, g)
end

# --- GPU ---------------------------------------------------------------------------

function test_gpu(d, xs)
    # JLArray is a CPU-backed GPUArray. It exercises the same broadcast machinery and
    # the same scalar-indexing ban as CUDA, so the real GPU failure modes are caught
    # on ordinary CI hardware with no device present.
    d32 = _withtype(d, Float32)
    x32 = Float32.(xs)
    expected = logdensityof.(d32, x32)

    @test isbits(d32)  # a non-isbits measure cannot be captured by a kernel

    # `allowscalar` only takes a do-block for *permitting* scalar indexing; to forbid
    # it you set the task-local flag. Leaving it off for the rest of the process is
    # the behaviour we want anyway.
    allowscalar(false)
    got = Array(logdensityof.(d32, JLArray(x32)))
    @test got ≈ expected
    @test eltype(got) === Float32
end
