# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Added

- `AbstractProbabilityMeasure{F,S}` and the normalized-only measure interface:
  `logdensityof`, `rand`, `support`, `insupport`, `params`, `checkparams`, and the
  moment/distribution-function surface.
- The `RealLine`, `NonNegativeReals`, `RealInterval`, `IntegerRange` and `RealVectors`
  supports. `PositiveReals` and `UnitInterval` will arrive with the first measure that
  needs one.
- `Normal(μ, σ)`, `Exponential(θ)` and `Uniform(a, b)`, with heterogeneous
  parameter types and no promotion or validation at construction.
- `MvNormal(μ, L)`, the first multivariate measure, with the
  `ContinuousMultivariateMeasure` alias it dispatches on. It is parameterized by the
  Cholesky factor rather than the covariance, which makes the reparameterized draw
  `μ + L z` plain arithmetic in the parameters and leaves the log-density one
  triangular solve instead of a factorization per evaluation. `cov` joins the
  re-exported summaries; there is no `cdf`, `quantile` or `median`, none of which has a
  closed form in more than one dimension.
- Fast paths for a `Diagonal` or `UniformScaling` factor, dispatched on the field type
  rather than given a measure type of their own. Whitening a general factor is an
  ``O(n^2)`` forward substitution; a structured one is a single elementwise division,
  and `checkparams` drops its sweep over off-diagonal zeros. A `UniformScaling` factor
  carries no dimension, taking ``n`` from `μ`, and is `isbits`.

  Both are the *factor*, so they carry standard deviations: `MvNormal(μ, σ * I)` is
  Distributions.jl's `MvNormal(μ, σ² * I)`. That divergence is the price of one
  consistent rule across all three forms, and it is called out in the docstring.
- `Categorical(p)`, the first discrete measure, with the `DiscreteUnivariateMeasure`
  alias it dispatches on. Its probabilities live in any `AbstractVector`, and its draws
  and quantiles are category indices in the float type those probabilities promote to,
  so that one code path serves AD and tracing backends. `checkparams` owns the
  sum-to-one requirement, as in Distributions.jl.
- `libs/ProbabilityMeasuresTest`: a reusable conformance suite (`test_measure`)
  covering interface conformance, totality, type genericity, type stability,
  zero allocations, normalization, cdf/quantile, moments, four AD backends, and
  GPU-shaped broadcast.
- ForwardDiff package extension, so reparameterized sampling is differentiable.

### Notes

- The conformance suite now flattens and unflattens parameters of any shape, so its AD
  and genericity blocks treat every measure as one parameter vector, and gates the
  blocks a measure cannot meet:

    - the allocation and moment blocks, written around a scalar draw, and the GPU
      broadcast run for univariate measures;
    - the `isbits` requirement runs for a measure whose parameters are all scalars;
    - normalization is a quadrature for a continuous measure and a summation for a
      discrete one;
    - the pathwise-derivative check runs for a measure whose draws are
      reparameterizable. A discrete draw is piecewise constant in the parameters, so it
      has none; its log-density gradient is still checked under every backend.

  A measure the defaults skip is still checked, in its own test file:
  `test/test-mvnormal.jl` carries a two-dimensional quadrature for normalization, a
  Monte Carlo check of the mean and covariance, and a bound on what the density
  allocates.
- `MvNormal`'s `logdensityof` allocates, unlike the univariate measures'. Whitening
  needs a temporary, grown by `vcat` so that reverse-mode backends, which reject array
  mutation, can differentiate it.
- The exported surface is intentionally minimal: every name is one a PPL is
  expected to call. `mode`, `skewness`, `kurtosis`, `mgf`, `cf`, `Matrixvariate`,
  `variateform`/`valuesupport`, and the unused supports are omitted rather than
  shipped speculatively, since adding an export later is non-breaking and removing
  one is not.

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/rsenne/ProbabilityMeasures.jl/compare/v0.1.0...HEAD
