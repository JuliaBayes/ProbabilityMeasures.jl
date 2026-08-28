# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Added

- `AbstractProbabilityMeasure{F,S}` and the normalized-only measure interface:
  `logdensityof`, `rand`, `support`, `insupport`, `params`, `checkparams`, `size` (the
  shape of a draw, `()` or `(n,)`), and the moment/distribution-function surface.
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
- `Geometric(p)`, counting failures before the first success. Every operation is closed
  form, entropy included, and `logcdf` and `logccdf` stay finite where `cdf` and `ccdf`
  underflow. `quantile` returns the smallest outcome whose CDF covers the probability,
  so `quantile(d, cdf(d, k))` recovers `k` until rounding makes the CDF reach one.
- `Weibull(α, θ)`, with shape `α` and scale `θ`. Every operation is closed form, the
  direct `logccdf` stays accurate in the upper tail, and the density at zero follows
  the shape: infinite below `α = 1`, `1/θ` at it, zero above. A sample is a
  transformed unit-exponential draw, differentiable in the parameters.
- `Gamma(α, θ)`, with a shape and a scale so that the mean is `α * θ`. Its density is
  closed form; `cdf`, `ccdf`, `logcdf`, `logccdf`, `quantile`, `median`, `entropy` and
  `rand` are not, and sum a series, run a continued fraction, or iterate Newton's method
  until the terms stop changing the result, in whatever type they are given, so
  `BigFloat` keeps its precision and differentiation tools follow the iteration. Sampling
  inverts the CDF, so the derivative of a draw with respect to either parameter is the
  exact reparameterization gradient, at the cost of a Newton solve per draw.
- `wrappedconditions(T)`, a trait for numbers whose comparisons are not a `Bool`. The
  Reactant extension sets it for traced numbers, and `Gamma`'s loops then run a fixed
  number of terms with `select` in place of every value-driven branch, so the same code
  compiles under Reactant. That path is exact for shapes up to about `900` in `Float64`
  and `2000` in `Float32`, and returns `NaN` above.
- `loggammap(a, x)` and `loggammaq(a, x)`, the regularized incomplete gamma integrals in
  log space, which keep `Gamma`'s `logcdf` and `logccdf` finite where the probabilities
  underflow. Relative accuracy sits at the rounding error of the argument type for shapes
  up to about `1000` and then falls off roughly in proportion to the shape: against
  `SpecialFunctions.gamma_inc` it is `5e-13` at shape `1000` and `2e-10` at shape `10^5`.
- `validateparams(d)`, which returns `d` or throws a `DomainError`, for the boundary
  where user-supplied parameters enter. It earns its place on `Categorical`, whose
  sum-to-one is the one invalid parameter a density cannot report: an unnormalized `p`
  gives a finite log-density, too large by `log(sum(p))`. It branches and throws, so it
  is the one exported name that cannot be traced.
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
- The conformance suite scalarizes a vector draw with a weighted sum rather than a
  plain one. A plain sum is one for every draw on a simplex, which left `Dirichlet`'s
  sample-derivative check with nothing to measure.
- The conformance suite recognizes structural parameters, those that set a measure's
  support or its loop lengths, such as `Binomial`'s `n`. They are held fixed rather
  than swept through the AD and element-type checks, which would otherwise ask for a
  dual-number or `Float32` trial count.
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
