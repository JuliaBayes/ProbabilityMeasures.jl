#=
  Shared conformance suite for ProbabilityMeasures.jl, in the same spirit as
  EmissionModels.jl's libs/EmissionModelsTest. Unregistered; the test suite
  `Pkg.develop`s it at runtime (see test/runtests.jl).

  Keeping it here rather than in the main package is what lets the conformance
  tooling -- JET, AllocCheck, four AD backends, JLArrays, QuadGK -- stay out of
  ProbabilityMeasures' dependency graph entirely. A measure library that a PPL
  depends on should be cheap to load.
=#
module ProbabilityMeasuresTest

using AllocCheck: check_allocs
using ConstructionBase: constructorof
using DensityInterface: densityof, logdensityof
using DifferentiationInterface: DifferentiationInterface
using DifferentiationInterface: AutoForwardDiff, AutoMooncake, AutoReverseDiff, AutoZygote
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using Interfaces: Interfaces, @implements, @interface
using JET: JET
using JLArrays: JLArray
using Mooncake: Mooncake
using ProbabilityMeasures
using QuadGK: quadgk
using Random: Xoshiro
using ReverseDiff: ReverseDiff
using Statistics: mean, median, quantile, std, var
using StatsAPI: params
using Test: @inferred, @test, @testset
using Zygote: Zygote

include("interface.jl")
include("conformance.jl")
include("implementations.jl")

export MeasureInterface
export test_measure, default_ad_backends, default_testpoints
export test_totality, test_genericity, test_inference, test_allocations
export test_normalization, test_cdf, test_moments, test_ad, test_gpu

end
