module ProbabilityMeasuresTest

using AllocCheck: AllocCheck, check_allocs
using ConstructionBase: constructorof
using DensityInterface: densityof, logdensityof
using DifferentiationInterface: DifferentiationInterface
using DifferentiationInterface: AutoForwardDiff, AutoMooncake, AutoReverseDiff, AutoZygote
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using GPUArraysCore: GPUArraysCore
using Interfaces: Interfaces, @implements, @interface
using JET: JET
using LinearAlgebra: Diagonal, I, UniformScaling
using JLArrays: JLArray
using Mooncake: Mooncake
using ProbabilityMeasures
using ProbabilityMeasures: ContinuousMeasure, DiscreteMeasure, UnivariateMeasure
using ProbabilityMeasures: DiagMvNormal, IsoMvNormal, unwhiten
using QuadGK: quadgk
using Random: Xoshiro
using ReverseDiff: ReverseDiff
using Statistics: cov, mean, median, quantile, std, var
using StatsAPI: params
using Test: @inferred, @test, @testset
using Zygote: Zygote

include("interface.jl")
include("conformance.jl")
include("implementations.jl")

export MeasureInterface
export test_measure, default_ad_backends, default_testpoints
export test_totality, test_genericity, test_inference, test_allocations
export test_normalization, test_cdf, test_moments, test_ad, test_gpu, test_reactant

end
