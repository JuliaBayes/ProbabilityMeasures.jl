# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Added

- `AbstractProbabilityMeasure{F,S}` and the normalized-only measure interface:
  `logdensityof`, `rand`, `support`, `insupport`, `params`, `checkparams`, and the
  moment/distribution-function surface.
- The `RealLine`, `NonNegativeReals`, `PositiveReals`, `RealInterval`, `IntegerRange`,
  `IntegerSimplex` and `RealVectors` supports. `UnitInterval` will arrive with the first
  measure that needs one.
- `Normal(μ, σ)`, `LogNormal(μ, σ)`, `Exponential(θ)` and `Uniform(a, b)`, with
  heterogeneous parameter types and no promotion or validation at construction.
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
- `Categorical(p)`, the first discrete measure. It accepts any `AbstractVector`, and
  `checkparams` validates that its probabilities are non-negative and sum to one.
- `Bernoulli(p)` and `Binomial(n, p)`. Samples carry the floating-point type of `p`, so
  `Bernoulli(0.5f0)` draws `Float32` zeros and ones rather than integers or booleans.
  `Binomial`'s `n` stays an `Integer`, since it sets both the support and the loop
  lengths. Its entropy, CDFs and quantile have no closed form and sum over the support
  in fixed-length loops, which keeps them traceable and GPU-safe, and each CDF tail is
  summed directly so a small tail is not lost to subtraction from one.
- `Laplace(μ, b)`, whose density has a kink at `x = μ`. `logcdf` and `logccdf`
  compute the near tail directly, so they stay finite where `cdf` and `ccdf` underflow,
  and the reparameterized draw is the difference of two exponential samples.
- `Cauchy(μ, σ)`, with direct stable formulas for both tails and a reparameterized
  inverse-CDF draw. Its undefined mean, variance, and standard deviation return `NaN`.
- `Multinomial(n, p)`, with the `DiscreteMultivariateMeasure` alias and count-vector
  support. Its density promotes the types of `p` and the count vector, while its
  moments, covariance and fixed-loop sampling preserve the numeric type of `p`.
- `Poisson(λ)` and its `NonNegativeIntegers` support. Density is constant time, as are
  CDFs, quantiles, and sampling for plain floating-point arguments, which use the
  regularized incomplete gamma. The wrapped numbers differentiation and tracing tools
  substitute fall back to a truncated sum that costs `O(λ)` and cannot run in traced or
  device-side code. Entropy always sums. Sampling uses CDF inversion to avoid underflow
  at large rates. These operations return `NaN` if the truncation bound exceeds `Int`.
- `validateparams(d)`, which returns `d` or throws a `DomainError`, for the boundary
  where user-supplied parameters enter. It earns its place on `Categorical`, whose
  sum-to-one is the one invalid parameter a density cannot report: an unnormalized `p`
  gives a finite log-density, too large by `log(sum(p))`. It branches and throws, so it
  is the one exported name that cannot be traced.
- `TDist(ν, μ, σ)`, Student's t with degrees of freedom, location and scale, and
  `TDist(ν)` for the standard form. `σ` is a scale rather than a standard deviation, so
  the variance is `σ²ν/(ν-2)`; `mean` is `NaN` at or below one degree of freedom and
  `var` is `Inf` between one and two. One degree of freedom reproduces `Cauchy` and a
  large one approaches `Normal`. The log-density uses `hypot` on the standardized value,
  which keeps it finite where squaring would overflow, and sampling inverts the CDF, so
  a draw and its quantile agree exactly.
- `MvTDist(ν, μ, L)`, the multivariate t, taking the same matrix, `Diagonal` and
  `UniformScaling` factors as `MvNormal`. `L * L'` is the scale matrix, not the
  covariance: the covariance is `ν/(ν-2) L L'` and exists only above two degrees of
  freedom. Sampling draws a direction on the sphere and a radius from the beta
  representation of the squared distance, which avoids the rejection loop a chi-square
  sampler would need and would not be traceable.
- `ProbabilityMeasures.betainc`, `betaincinv` and `logbetainc`, the regularized
  incomplete beta, its inverse, and its logarithm, on which both t measures rest.
  SpecialFunctions provides the first two only for `Float16`, `Float32` and `Float64`,
  and the conformance suite checks distribution functions in `BigFloat`. `betainc` sums
  the Abramowitz & Stegun continued fraction by Lentz's method; `betaincinv` runs
  Newton's method in `log(x)` inside a bracket. Both pass an argument and its complement
  as a pair, so neither tail is recovered by subtracting the other from one, and
  `logbetainc` takes the logarithm before the exponential, which is what keeps `logcdf`
  and `logccdf` finite where the tails themselves underflow.
- `ProbabilityMeasures.loggammat` and `sqrtt`, which return `NaN` where `loggamma` and
  `sqrt` throw, so an invalid `ν` reaches the density as a non-finite value.
- `libs/ProbabilityMeasuresTest`: a reusable conformance suite (`test_measure`)
  covering interface conformance, totality, type genericity, type stability,
  zero allocations, normalization, cdf/quantile, moments, four AD backends, and
  GPU-shaped broadcast.
- ForwardDiff package extension, so reparameterized sampling is differentiable.

### Notes

- The conformance suite now supports array parameters and discrete measures. It
  flattens parameters for AD checks, sums discrete probability masses for normalization,
  and skips pathwise sampling derivatives for discrete draws.
- Discrete normalization tests now skip supports without a finite maximum. `Poisson`
  provides its own normalization test.
- The conformance allocation check now keeps every allocation but only the dynamic
  dispatch in this package's own code. `gamma_inc` reaches a `ccall` wrapper in
  SpecialFunctions that AllocCheck cannot resolve and that allocates nothing.
- The conformance suite recognizes structural parameters, those that set a measure's
  support or its loop lengths, such as `Binomial`'s `n`. They are held fixed rather
  than swept through the AD and element-type checks, which would otherwise ask for a
  dual-number or `Float32` trial count.
- The two t measures reach the incomplete beta only for plain floating-point types,
  which is what `cdf`, `ccdf`, `quantile` and `rand` need. The wrapped numbers
  differentiation and tracing tools substitute have no method, so a draw carries
  derivatives in the location and scale but not in the degrees of freedom, and a
  `MethodError` says so rather than a silently zero gradient. `logdensityof` and
  `entropy` are closed forms and differentiate in every parameter.
- `MvNormal` and `MvTDist` share an internal `MvLocationScale` supertype holding the
  whitening, log-determinant, shape-check and scale-matrix code. The three factor forms
  dispatch on the factor itself rather than on the measure, so a new location-scale
  measure gets all three by declaring its supertype.
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
