using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, I, LowerTriangular, diag
using QuadGK: quadgk
using Random: Xoshiro
using Test

# Build the reference scale matrix from the lower triangle of `L`.
function lowertriangle(d)
    n = length(d.μ)
    return [i >= j ? Float64(d.L[i, j]) : 0.0 for i in 1:n, j in 1:n]
end

function reference(d)
    L = lowertriangle(d)
    return Distributions.MvTDist(Float64(d.ν), Float64.(d.μ), L * L')
end

reference_logpdf(d, x) = Distributions.logpdf(reference(d), collect(float.(x)))

const CORRELATED = MvTDist(8.0, [1.0, -2.0], [2.0 0.0; 0.5 1.5])

@testset "conformance" begin
    ds = (
        MvTDist(6.0, [0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        CORRELATED,
        MvTDist(6.0f0, Float32[0.0, 1.0], Float32[1.0 0.0; -0.25 0.5]),
        MvTDist(5, [0, 0, 0], [2 0 0; 1 2 0; 0 1 2]),
        MvTDist(7.0, [1.0, -2.0], Diagonal([2.0, 1.5])),
        MvTDist(5.0, [1.0, -2.0], 1.5 * I),
        MvTDist(9, [0, 0, 0], 2 * I),
    )
    for (i, d) in enumerate(ds)
        test_measure(d; name="MvTDist $i", reference_logpdf=reference_logpdf)
    end
end

@testset "reference numerics against Distributions.jl" begin
    ds = (
        MvTDist(6.0, [0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        CORRELATED,
        MvTDist(4.5, [2.0, -1.0, 0.5], [1.5 0.0 0.0; -0.5 0.75 0.0; 0.25 1.0 2.0]),
    )
    for d in ds
        r = reference(d)
        n = length(d.μ)
        for x in (zeros(n), d.μ, d.μ .+ 1, d.μ .- 3.5, 4 .* ones(n))
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

@testset "one dimension matches the univariate measure" begin
    d, u = MvTDist(6.0, [1.0], reshape([2.0], 1, 1)), TDist(6.0, 1.0, 2.0)
    for x in (-3.0, 1.0, 7.5, 1e8)
        @test logdensityof(d, [x]) ≈ logdensityof(u, x)
    end
    @test only(mean(d)) == mean(u)
    @test only(var(d)) ≈ var(u)
    @test entropy(d) ≈ entropy(u)
end

@testset "many degrees of freedom approach the normal" begin
    d = MvTDist(1.0e8, [1.0, -2.0], [2.0 0.0; 0.5 1.5])
    n = MvNormal([1.0, -2.0], [2.0 0.0; 0.5 1.5])
    for x in ([0.3, -1.0], [1.0, -2.0], [4.0, 4.0])
        @test logdensityof(d, x) ≈ logdensityof(n, x) atol = 1e-5
    end
    @test cov(d) ≈ cov(n) rtol = 1e-7
    @test entropy(d) ≈ entropy(n) atol = 1e-6
end

@testset "the scale matrix is not the covariance" begin
    d = MvTDist(6.0, [0.0, 0.0], Diagonal([2.0, 1.5]))
    scale = Diagonal([4.0, 2.25])
    @test cov(d) ≈ (6 / 4) .* scale
    @test var(d) ≈ (6 / 4) .* diag(scale)
    @test cov(d) isa Diagonal

    # Below three degrees of freedom the second moment diverges, and below two the
    # first one does too.
    @test all(isinf, var(MvTDist(2.0, [0.0, 0.0], Diagonal([2.0, 1.5]))))
    @test all(isinf, var(MvTDist(1.5, [0.0, 0.0], Diagonal([2.0, 1.5]))))
    @test all(isnan, var(MvTDist(1.0, [0.0, 0.0], Diagonal([2.0, 1.5]))))
    @test all(isnan, mean(MvTDist(1.0, [3.0, 4.0], Diagonal([2.0, 1.5]))))
    @test mean(MvTDist(1.5, [3.0, 4.0], Diagonal([2.0, 1.5]))) == [3.0, 4.0]
end

@testset "normalization by two-dimensional quadrature" begin
    for d in (MvTDist(6.0, [0.0, 0.0], [1.0 0.0; 0.0 1.0]), CORRELATED)
        # The tails are polynomial, so integrate over the whole plane rather than a
        # box of standard deviations.
        inner(x1) = first(quadgk(x2 -> densityof(d, [x1, x2]), -Inf, Inf; rtol=1e-10))
        total, err = quadgk(inner, -Inf, Inf; rtol=1e-9)
        @test total ≈ 1 atol = max(1e-7, 10err)
    end
end

@testset "moments against Monte Carlo" begin
    d, nsamples = CORRELATED, 400_000
    samples = reduce(hcat, rand(Xoshiro(20250801), d, nsamples))

    m = vec(sum(samples; dims=2)) ./ nsamples
    @test m ≈ mean(d) atol = 8 * maximum(std(d)) / sqrt(nsamples)

    centered = samples .- m
    @test centered * centered' ./ (nsamples - 1) ≈ cov(d) rtol = 0.05
end

@testset "structured factors agree with the general path" begin
    densefactor(v, n) = [i == j ? v[i] : 0.0 for i in 1:n, j in 1:n]

    for n in (2, 3, 5)
        μ = [1.5 - i for i in 1:n]
        σ = [1.0 + i / 4 for i in 1:n]
        λ, ν = 1.5, 6.0
        pairs = (
            (MvTDist(ν, μ, Diagonal(σ)), MvTDist(ν, μ, densefactor(σ, n))),
            (MvTDist(ν, μ, λ * I), MvTDist(ν, μ, densefactor(fill(λ, n), n))),
        )
        for (structured, full) in pairs
            for x in (zeros(n), μ, μ .+ 0.3, μ .- 4.0)
                @test logdensityof(structured, x) ≈ logdensityof(full, x)
            end
            @test cov(structured) ≈ cov(full)
            @test var(structured) ≈ var(full)
            @test entropy(structured) ≈ entropy(full)
            @test checkparams(structured) == checkparams(full)
            @test support(structured) === support(full)
            @test rand(Xoshiro(1), structured) ≈ rand(Xoshiro(1), full)
            @test isnan(logdensityof(structured, zeros(n - 1)))
        end
    end
end

@testset "construction never validates" begin
    μ, L = [0.0, 0.0], [1.0 0.0; 0.0 1.0]
    for bad in (
        MvTDist(0.0, μ, L),
        MvTDist(-1.0, μ, L),
        MvTDist(NaN, μ, L),
        MvTDist(Inf, μ, L),
        MvTDist(6.0, μ, [1.0 0.0; 0.0 0.0]),
        MvTDist(6.0, μ, [1.0 0.0; 0.0 -1.0]),
        MvTDist(6.0, [Inf, 0.0], L),
        MvTDist(6.0, μ, [1.0 0.0 0.0; 0.0 1.0 0.0]),
        MvTDist(6.0, μ, Diagonal([1.0, 1.0, 1.0])),
        MvTDist(6.0, μ, 0.0 * I),
    )
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, [0.5, 0.5]))
    end
    @test checkparams(CORRELATED)
end

@testset "density handles wrong argument lengths" begin
    d = CORRELATED
    @test isnan(logdensityof(d, [0.5]))
    @test isnan(logdensityof(d, Float64[]))
    @test isnan(logdensityof(d, [0.5, 0.5, 0.5]))
    for x in ([Inf, Inf], [-Inf, 0.0], [NaN, 0.0], fill(floatmax(Float64), 2))
        @test !isfinite(logdensityof(d, x))
    end
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(0.0, 1.0)
    L = [1.0 0.0; 0.0 1.0]
    @test typeof(MvTDist(6.0, [dual, dual], L)) ===
        MvTDist{Float64,Vector{typeof(dual)},typeof(L)}
    @test typeof(MvTDist(6, Float32[0, 0], L)) ===
        MvTDist{Int,Vector{Float32},Matrix{Float64}}
    @test MvTDist(6.0, [0.0, 0.0], LowerTriangular(L)).L isa LowerTriangular
end

@testset "precision follows the argument, not the parameters" begin
    exact = MvTDist(5, [0, 0], [2 0; 1 2])
    @test logdensityof(exact, Float32[1, 2]) isa Float32
    @test logdensityof(exact, big.([1.0, 2.0])) isa BigFloat

    x = big.([1.0, 2.0])
    full = logdensityof(MvTDist(big(5), big.([0, 0]), big.([2 0; 1 2])), x)
    @test abs(logdensityof(exact, x) - full) < 1e-70

    rational = MvTDist(5//1, [0//1, 0//1], [1//1 0//1; 1//2 3//2])
    afloat = MvTDist(5.0, [0.0, 0.0], [1.0 0.0; 0.5 1.5])
    @test checkparams(rational)
    for x in ([1//3, 1//5], [2//7, -3//4], [0//1, 0//1])
        v = logdensityof(rational, x)
        @test v isa Float64
        @test v ≈ logdensityof(afloat, float.(x))
    end
end

@testset "sampling" begin
    d = CORRELATED
    @test rand(Xoshiro(1), d) isa Vector{Float64}
    @test rand(Xoshiro(1), MvTDist(6.0f0, Float32[0, 0], Float32[1 0; 0 1])) isa
        Vector{Float32}
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Vector{Float64}
    @test size(d) == (2,)

    # A draw is `μ + L z` for a radially scaled `z`, so the derivative in `μ` is the
    # identity and a factor entry's derivative is its share of the scaled noise.
    jac = ForwardDiff.jacobian(m -> rand(Xoshiro(7), MvTDist(d.ν, m, d.L)), d.μ)
    @test jac == [1.0 0.0; 0.0 1.0]
    scaled = rand(Xoshiro(7), MvTDist(d.ν, zeros(2), [1.0 0.0; 0.0 1.0]))
    secondrow = a -> rand(Xoshiro(7), MvTDist(d.ν, d.μ, [2.0 0.0; a 1.5]))[2]
    @test ForwardDiff.derivative(secondrow, 0.5) ≈ scaled[1]
end

@testset "support" begin
    d = CORRELATED
    @test support(d) === RealVectors(2)
    @test insupport(d, [0.0, 0.0])
    @test !insupport(d, [0.0])
    @test !insupport(d, [Inf, 0.0])
end

@testset "show" begin
    @test string(CORRELATED) == "MvTDist(ν=8.0, μ=[1.0, -2.0], L=[2.0 0.0; 0.5 1.5])"
end
