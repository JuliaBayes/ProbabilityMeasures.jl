# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Added

- `AbstractProbabilityMeasure{F,S}` and the normalized-only measure interface:
  `logdensityof`, `rand`, `support`, `insupport`, `params`, `checkparams`,
  `reference`, and the moment/distribution-function surface.
- `Lebesgue` and `Counting` reference measures, derived automatically from the
  `ValueSupport` type parameter.
- `RealLine`, `PositiveReals`, `UnitInterval` and `RealInterval` supports.
- `Normal(μ, σ)`, with heterogeneous parameter types and no promotion or
  validation at construction.
- `libs/ProbabilityMeasuresTest`: a reusable conformance suite (`test_measure`)
  covering interface conformance, totality, type genericity, type stability,
  zero allocations, normalization, cdf/quantile, moments, four AD backends, and
  GPU-shaped broadcast.
- ForwardDiff package extension, so reparameterized sampling is differentiable.

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/rsenne/ProbabilityMeasures.jl/compare/v0.1.0...HEAD
