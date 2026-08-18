module ProbabilityMeasuresTest

using AllocCheck: check_allocs
using ConstructionBase: constructorof
using DensityInterface: densityof, logdensityof
using DifferentiationInterface: DifferentiationInterface
using DifferentiationInterface: AutoForwardDiff, AutoMooncake, AutoReverseDiff, AutoZygote
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using GPUArraysCore: GPUArraysCore
using Interfaces: Interfaces, @implements, @interface
using JET: JET
#= Structured parameters flatten to fewer numbers than their shape suggests. =#
using LinearAlgebra: Diagonal, I, UniformScaling
using JLArrays: JLArray
using Mooncake: Mooncake
using ProbabilityMeasures
#=
  Unexported dispatch aliases. The conditional `check_*` defaults ask what kind of
  measure `d` is, and `isa ContinuousMeasure` is the idiom the package itself uses.
=#
using ProbabilityMeasures: ContinuousMeasure, UnivariateMeasure
#=
  `unwhiten` maps standard normal coordinates back, which is how `MvNormal`'s default test
  points are built. The two aliases are what the structured factors dispatch on, so that
  the per-measure hooks reach the same specialized paths the measure itself does.
=#
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
