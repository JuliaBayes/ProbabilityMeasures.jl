using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, I, LowerTriangular, cholesky, diag
using QuadGK: quadgk
using Random: Xoshiro
using Test

#=
  Distributions.jl is a test-only numerical reference. `MvNormal` is parameterized by
  the Cholesky factor, so the reference takes `L L'` as its covariance.

  The factor is read entry by entry, and only its lower triangle, which is both what the
  measure itself promises and the one spelling that covers a matrix, a `Diagonal` and a
  `UniformScaling` alike: the last of these is not an array and does not broadcast.
=#
function lowertriangle(m)
    n = length(m.μ)
    return [i >= j ? Float64(m.L[i, j]) : 0.0 for i in 1:n, j in 1:n]
end

function reference(m)
    L = lowertriangle(m)
    return Distributions.MvNormal(Float64.(m.μ), L * L')
end

reference_logpdf(m, x) = Distributions.logpdf(reference(m), collect(float.(x)))

const CORRELATED = MvNormal([1.0, -2.0], [2.0 0.0; 0.5 1.5])

@testset "conformance" begin
    ds = (
        MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        CORRELATED,
        MvNormal(Float32[0.0, 1.0], Float32[1.0 0.0; -0.25 0.5]),
        MvNormal([0, 0, 0], [2 0 0; 1 2 0; 0 1 2]),
        # The structured factors take their own paths through every one of these blocks.
        MvNormal([1.0, -2.0], Diagonal([2.0, 1.5])),
        MvNormal(Float32[0.0, 1.0], Diagonal(Float32[0.5, 2.0])),
        MvNormal([1.0, -2.0], 1.5 * I),
        MvNormal([0, 0, 0], 2 * I),
    )
    for (i, d) in enumerate(ds)
        test_measure(d; name="MvNormal $i", reference_logpdf=reference_logpdf)
    end
end

#=
  The blocks `test_measure` skips for a multivariate measure, done here in the shape a
  multivariate measure needs: `test_normalization` integrates in one dimension,
  `test_moments` compares scalar summaries, `test_allocations` proves a scalar draw
  allocates nothing, and `test_gpu` broadcasts over a device array of scalars.
=#

@testset "normalization by two-dimensional quadrature" begin
    for d in (MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), CORRELATED)
        #=
          Ten marginal standard deviations leave less than 1e-22 outside the box, well
          under the tolerance the quadrature itself reaches.
        =#
        lo, hi = mean(d) .- 10 .* std(d), mean(d) .+ 10 .* std(d)
        inner(x1) = first(quadgk(x2 -> densityof(d, [x1, x2]), lo[2], hi[2]; rtol=1e-11))
        total, err = quadgk(inner, lo[1], hi[1]; rtol=1e-10)
        @test total ≈ 1 atol = max(1e-8, 10err)
    end
end

@testset "moments against Monte Carlo" begin
    d, nsamples = CORRELATED, 200_000
    draws = rand(Xoshiro(20250801), d, nsamples)
    samples = reduce(hcat, draws)

    m = vec(sum(samples; dims=2)) ./ nsamples
    # Monte Carlo error on each coordinate mean is std/sqrt(n); allow five of them.
    @test m ≈ mean(d) atol = 5 * maximum(std(d)) / sqrt(nsamples)

    centered = samples .- m
    @test centered * centered' ./ (nsamples - 1) ≈ cov(d) rtol = 20 / sqrt(nsamples)
end

@testset "the density allocates, but only a whitened point" begin
    d, x = CORRELATED, [0.3, -1.0]
    logdensityof(d, x)                         # compile before measuring
    bytes = @allocated logdensityof(d, x)
    @test bytes == @allocated logdensityof(d, x)
    #=
      Forward substitution grows the whitened vector one row at a time, so the count is
      one small array per dimension and nothing else. The bound is loose enough not to
      depend on the allocator's minimum block, tight enough to fail if the density ever
      starts allocating per evaluation point.
    =#
    @test bytes <= 1024
end

@testset "reference numerics against Distributions.jl" begin
    ds = (
        MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        CORRELATED,
        MvNormal([2.0, -1.0, 0.5], [1.5 0.0 0.0; -0.5 0.75 0.0; 0.25 1.0 2.0]),
    )
    for d in ds
        r = reference(d)
        n = length(mean(d))
        for x in (zeros(n), mean(d), mean(d) .+ 1, mean(d) .- 3.5, 4 .* ones(n))
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, x)
            @test densityof(d, x) ≈ Distributions.pdf(r, x)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test cov(d) ≈ Distributions.cov(r)
        @test var(d) ≈ diag(Distributions.cov(r))
        @test std(d) ≈ sqrt.(diag(Distributions.cov(r)))
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

#=
  The structured factors are an optimization, so the thing to test is that they change
  nothing observable. Each is compared against the same measure written as a full matrix,
  which takes the general path.
=#
@testset "structured factors agree with the general path" begin
    μ = [1.0, -2.0]
    pairs = (
        (MvNormal(μ, Diagonal([2.0, 1.5])), MvNormal(μ, [2.0 0.0; 0.0 1.5])),
        (MvNormal(μ, 1.5 * I), MvNormal(μ, [1.5 0.0; 0.0 1.5])),
    )
    for (structured, full) in pairs
        for x in ([0.0, 0.0], [0.3, -1.0], mean(structured), [-4.0, 7.5])
            @test logdensityof(structured, x) == logdensityof(full, x)
            @test densityof(structured, x) == densityof(full, x)
        end
        @test cov(structured) == cov(full)
        @test var(structured) == var(full)
        @test std(structured) == std(full)
        @test mean(structured) == mean(full)
        @test entropy(structured) ≈ entropy(full)
        @test checkparams(structured) == checkparams(full)
        @test support(structured) === support(full)
        # Same reparameterization, so the same seed gives the same draw.
        @test rand(Xoshiro(1), structured) == rand(Xoshiro(1), full)
        # And the same totality in the shape of the argument.
        @test isnan(logdensityof(structured, [0.5]))
    end

    #=
      A structured factor has its own `checkparams` and its own whitening, so it needs its
      own invalid cases: a zero diagonal entry divides by zero, a negative one takes the
      log of a negative number.
    =#
    for bad in (
        MvNormal(μ, Diagonal([0.0, 1.5])),
        MvNormal(μ, Diagonal([-1.0, 1.5])),
        MvNormal(μ, 0.0 * I),
        MvNormal(μ, -1.5 * I),
        MvNormal(μ, Diagonal([1.0, 1.0, 1.0])),
        MvNormal([Inf, 0.0], 1.5 * I),
    )
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, [0.5, 0.5]))
    end
end

@testset "structured factors keep their structure" begin
    diagonal = MvNormal([1.0, -2.0], Diagonal([2.0, 1.5]))
    isotropic = MvNormal([1.0, -2.0], 1.5 * I)

    # A structured factor squares to a structured covariance rather than a dense one.
    @test cov(diagonal) isa Diagonal
    @test cov(isotropic) isa Diagonal

    #=
      An isotropic factor carries no dimension, so `n` comes from `μ`, and it is `isbits`
      where a matrix-backed factor is not.
    =#
    @test support(isotropic) === RealVectors(2)
    @test isbits(isotropic.L)
    @test !isbits(MvNormal([1.0, -2.0], [1.5 0.0; 0.0 1.5]).L)

    #=
      The second argument is the factor, so these are standard deviations. Distributions.jl
      spells the same measure with a variance.
    =#
    @test var(isotropic) ≈ fill(1.5^2, 2)
    @test var(diagonal) ≈ [2.0^2, 1.5^2]
    @test cov(isotropic) ≈ Distributions.cov(Distributions.MvNormal([1.0, -2.0], 1.5^2 * I))
end

@testset "a covariance is factored once by the caller" begin
    Σ = [4.0 1.0; 1.0 2.5]
    d = MvNormal([1.0, -2.0], Matrix(cholesky(Σ).L))
    @test cov(d) ≈ Σ
    @test logdensityof(d, [0.3, -1.0]) ≈
        Distributions.logpdf(Distributions.MvNormal([1.0, -2.0], Σ), [0.3, -1.0])
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(0.0, 1.0)
    L = [1.0 0.0; 0.0 1.0]
    @test typeof(MvNormal([dual, dual], L)) === MvNormal{Vector{typeof(dual)},typeof(L)}
    @test typeof(MvNormal(Float32[0, 0], L)) === MvNormal{Vector{Float32},Matrix{Float64}}

    # A Float32 parameter meeting a Float64 literal must not silently widen.
    @test MvNormal(Float32[0, 0], Float32.(L)).L isa Matrix{Float32}
    # Any indexable matrix serves as the factor, and it keeps its own type.
    @test MvNormal([0.0, 0.0], LowerTriangular(L)).L isa LowerTriangular
end

@testset "precision follows the argument, not the parameters" begin
    exact = MvNormal([0, 0], [2 0; 1 2])
    @test logdensityof(exact, Float32[1, 2]) isa Float32
    @test logdensityof(exact, big.([1.0, 2.0])) isa BigFloat

    #=
      Check that no Float64 intermediate caps BigFloat precision.
    =#
    x = big.([1.0, 2.0])
    full = logdensityof(MvNormal(big.([0, 0]), big.([2 0; 1 2])), x)
    @test abs(logdensityof(exact, x) - full) < 1e-70
end

@testset "construction never validates" begin
    μ = [0.0, 0.0]
    # A zero diagonal entry: the solve divides by it.
    @test !checkparams(MvNormal(μ, [1.0 0.0; 0.0 0.0]))
    # A negative one: the log of it is not a number.
    @test !checkparams(MvNormal(μ, [1.0 0.0; 0.0 -1.0]))
    @test !checkparams(MvNormal([Inf, 0.0], [1.0 0.0; 0.0 1.0]))
    # Shapes that do not line up, including the empty measure.
    @test !checkparams(MvNormal(μ, [1.0 0.0 0.0; 0.0 1.0 0.0]))
    @test !checkparams(MvNormal(μ, [1.0 0.0; 0.0 1.0; 0.0 0.0]))
    @test !checkparams(MvNormal(Float64[], zeros(0, 0)))
    @test checkparams(CORRELATED)

    for bad in (
        MvNormal(μ, [1.0 0.0; 0.0 0.0]),
        MvNormal(μ, [1.0 0.0; 0.0 -1.0]),
        MvNormal([Inf, 0.0], [1.0 0.0; 0.0 1.0]),
        MvNormal(μ, [1.0 0.0 0.0; 0.0 1.0 0.0]),
    )
        @test !isfinite(logdensityof(bad, [0.5, 0.5]))
    end
end

@testset "the density is total in the shape of its argument" begin
    d = CORRELATED
    @test isnan(logdensityof(d, [0.5]))
    @test isnan(logdensityof(d, Float64[]))
    @test isnan(logdensityof(d, [0.5, 0.5, 0.5]))
    # The `NaN` still carries the type the promotion invariant asks for.
    @test logdensityof(d, Float32[0.5]) isa Float64
    @test logdensityof(MvNormal(Float32[0, 0], Float32[1 0; 0 1]), Float32[0.5]) isa Float32
    for x in ([Inf, Inf], [-Inf, 0.0], [NaN, 0.0], fill(floatmax(Float64), 2))
        @test !isfinite(logdensityof(d, x))
    end
end

@testset "only the lower triangle of the factor is read" begin
    d = CORRELATED
    littered = MvNormal(d.μ, [2.0 7.0; 0.5 1.5])
    x = [0.3, -1.0]
    @test logdensityof(littered, x) == logdensityof(d, x)
    @test cov(littered) == cov(d)
    @test rand(Xoshiro(1), littered) == rand(Xoshiro(1), d)
end

@testset "support" begin
    d = CORRELATED
    @test support(d) === RealVectors(2)
    @test insupport(d, [0.0, 0.0])
    @test !insupport(d, [0.0])
    @test !insupport(d, [0.0, 0.0, 0.0])
    @test !insupport(d, [Inf, 0.0])
    @test !insupport(d, [NaN, 0.0])
    @test isbits(RealVectors(2))
end

@testset "sampling" begin
    d = CORRELATED
    @test rand(Xoshiro(1), d) isa Vector{Float64}
    @test rand(Xoshiro(1), MvNormal(Float32[0, 0], Float32[1 0; 0 1])) isa Vector{Float32}
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Vector{Float64}

    #=
      The draw is the reparameterization, exactly: noise drawn in the plain float type,
      then pushed through `μ + L z`.
    =#
    z = randn(Xoshiro(7), Float64, 2)
    @test rand(Xoshiro(7), d) == d.μ + d.L * z

    #=
      So its Jacobian in `μ` is the identity, and its derivative in a factor entry is
      the noise that entry multiplies.
    =#
    jac = ForwardDiff.jacobian(m -> rand(Xoshiro(7), MvNormal(m, d.L)), d.μ)
    @test jac == [1.0 0.0; 0.0 1.0]
    secondrow = a -> rand(Xoshiro(7), MvNormal(d.μ, [2.0 0.0; a 1.5]))[2]
    @test ForwardDiff.derivative(secondrow, 0.5) == z[1]
end

@testset "show" begin
    @test string(CORRELATED) == "MvNormal(μ=[1.0, -2.0], L=[2.0 0.0; 0.5 1.5])"
end
