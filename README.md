# ProbabilityMeasures.jl

[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/dev)
[![Tests](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl)

ProbabilityMeasures.jl provides normalized probability measures designed for use in
probabilistic programs. Its implementations are type-generic, allocation-free in core
density and sampling operations, and compatible with automatic differentiation,
broadcasting on GPU arrays, and Reactant tracing.

The package is experimental. At present it implements `Normal`, `LogNormal`,
`Exponential`, `Weibull`, `Gamma`, `Beta`, `Uniform`, `Laplace`, `Cauchy`,
`Categorical`, `Bernoulli`, `Binomial`, `Poisson`, `Geometric`, `MvNormal`, `Dirichlet`,
and `Multinomial`.

## Installation

Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/rsenne/ProbabilityMeasures.jl")
```

## Quick start

```julia
using ProbabilityMeasures

d = Normal(1.0, 2.0)

logdensityof(d, 0.5)
densityof(d, 0.5)
cdf(d, 0.5)
quantile(d, 0.95)

mean(d)
var(d)
rand(d)
```

Constructors preserve parameter types and do not validate:

```julia
julia> typeof(Normal(0.0f0, 1))
Normal{Float32, Int64}

julia> d = Normal(0.0, -1.0);

julia> checkparams(d)
false

julia> logdensityof(d, 0.0)
NaN
```

Use `checkparams` when accepting user-supplied parameters. Density evaluation itself
is total: invalid parameters and unsupported values return non-finite results rather
than throwing.

## Available API

`Normal(μ, σ)`, `LogNormal(μ, σ)`, `Exponential(θ)`, `Weibull(α, θ)`, `Gamma(α, θ)`,
`Beta(α, β)`, `Uniform(a, b)`, `Laplace(μ, b)`, `Cauchy(μ, σ)`, `Categorical(p)`,
`Bernoulli(p)`, `Binomial(n, p)`, `Poisson(λ)`, and `Geometric(p)` each support:

- `densityof` and `logdensityof`
- `cdf`, `ccdf`, `logcdf`, and `logccdf`
- `quantile`, `mean`, `median`, `var`, `std`, and `entropy`
- `rand` and Random's array-sampling methods
- `params`, `support`, `insupport`, and `checkparams`

`Cauchy(μ, σ)` has no finite mean or variance, so `mean`, `var` and `std` return `NaN`.
`median` and `entropy` are exact.

`Gamma(α, θ)` takes a shape and a scale, so its mean is `α * θ`, and `Gamma(α)` sets the
scale to one. Its density is closed form, but `cdf`, `ccdf`, `logcdf`, `logccdf`,
`quantile`, `median`, `entropy` and `rand` are not. They iterate until the terms stop
changing the result, so `BigFloat` keeps its precision and differentiation tools follow
the iteration. Under Reactant they run a fixed number of terms instead, exact for shapes
up to about `900` in `Float64`. Sampling inverts the CDF, so derivatives of a draw with
respect to `α` and `θ` are exact.

`Beta(α, β)` has a closed-form density. Its `cdf`, `ccdf`, `logcdf`, `logccdf`,
`quantile`, `median` and `rand` run a continued fraction and Newton's method until the
terms stop changing the result, or a fixed number of terms under Reactant. A draw is
`X / (X + Y)` for two `Gamma` draws, so its derivatives with respect to `α` and `β` are
exact.

`Categorical(p)` assigns the probabilities in `p` to categories `1:length(p)`. Draws
and quantiles use the promoted floating-point type of `p`:

```julia
julia> d = Categorical([0.2, 0.3, 0.5]);

julia> quantile(d, 0.5)
2.0

julia> rand(d) isa Float64
true

julia> logdensityof(d, 2.0), logdensityof(d, 2.5)
(-1.2039728043259361, -Inf)
```

`MvNormal(μ, L)` takes a lower-triangular covariance factor, so `cov(d) == L * L'`.
If you have a covariance matrix, factor it before constructing the measure.

```julia
using LinearAlgebra, ProbabilityMeasures

Σ = [4.0 1.0; 1.0 2.5]
d = MvNormal([1.0, -2.0], Matrix(cholesky(Σ).L))

logdensityof(d, [0.3, -1.0])
mean(d), cov(d), var(d), std(d), entropy(d)
rand(d)
```

It supports `densityof`, `logdensityof`, `rand`, `mean`, `cov`, `var`, `std`, `entropy`,
`params`, `support`, `insupport`, and `checkparams`. `var` and `std` return marginal
values. Multivariate `cdf`, `quantile`, and `median` are not provided.

Diagonal and isotropic factors are also supported:

```julia
MvNormal(μ, L)                # general
MvNormal(μ, Diagonal(σ))      # independent coordinates, σ their standard deviations
MvNormal(μ, σ * I)            # isotropic, σ the common standard deviation
```

The second argument is always a factor. In the diagonal and isotropic forms, `σ`
contains standard deviations, not variances.

`Multinomial(n, p)` supports `densityof`, `logdensityof`, `rand`, `mean`, `cov`, `var`,
`std`, `params`, `support`, `insupport`, and `checkparams`. Its samples are count vectors
in `IntegerSimplex(n, length(p))`, and `var` and `std` return marginal values. As with
`MvNormal`, multivariate `cdf`, `quantile`, and `median` are not provided.

`Dirichlet(α)` supports `densityof`, `logdensityof`, `rand`, `mean`, `cov`, `var`,
`std`, `entropy`, `params`, `support`, `insupport`, and `checkparams`. Its samples are
probability vectors in `RealSimplex(length(α))`, and because a draw sums to one, the
density is the one with respect to the first `length(α) - 1` entries, matching
Distributions.jl. A draw is a vector of `Gamma` draws divided by their sum, so it is
differentiable in every shape. Multivariate `cdf`, `quantile`, and `median` are not
provided.

The density result follows normal Julia promotion rules across the parameters and
evaluation point:

```julia
julia> logdensityof(Normal(0, 1), 1.0f0) isa Float32
true

julia> logdensityof(Normal(0, 1), big"1.0") isa BigFloat
true
```

## Broadcasting and accelerators

Measures broadcast as scalars. Use broadcasting for batched density evaluation:

```julia
d = Normal(0.0f0, 1.0f0)
xs = randn(Float32, 1024)
ys = logdensityof.(d, xs)
```

The same form works with GPU array types whose broadcast implementation supports the
operations involved. Package extensions provide integration with ForwardDiff,
ReverseDiff, EnzymeCore, and Reactant.

```julia
using ProbabilityMeasures, Reactant

d = Normal(0.0, 1.0)
xs = Reactant.to_rarray(randn(1000))
@jit logdensityof.(d, xs)
```

Scalar sampling is reparameterized: noise is drawn in the underlying floating-point
type and the measure parameters enter through arithmetic. This allows derivatives with
respect to the parameters without custom derivative rules. Categorical draws do not
have a pathwise derivative, but their log-density is differentiable with respect to
the probabilities.

`Categorical` accepts any `AbstractVector`. Use an `isbits` vector type, such as
`StaticArrays.SVector`, when the complete measure must be `isbits`.

## Defining a measure

A subtype of `AbstractProbabilityMeasure` must implement:

- `DensityInterface.logdensityof(d, x)`
- `Base.rand(rng::AbstractRNG, d)`
- `Base.eltype(::Type{typeof(d)})`
- `Base.size(d)` (multivariate measures only; univariate measures inherit `()`)
- `support(d)`

Implementations must also keep `logdensityof` total, avoid constructor validation,
and support numeric wrapper types used by AD and tracing systems. See the
[developer documentation](https://rsenne.github.io/ProbabilityMeasures.jl/dev/91-developer/)
for the complete contract.

The reusable conformance suite lives in `libs/ProbabilityMeasuresTest`:

```julia
using ProbabilityMeasuresTest

test_measure(Normal(0.0, 1.0))
```

It checks the interface, normalization, numerical behavior, type stability,
allocations, automatic differentiation, GPU-style broadcasting, and Reactant tracing.

## Development

Instantiate and run the test environment from the repository root:

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
```

See the [contribution guide](docs/src/90-contributing.md) for contribution guidelines.

## Current scope

ProbabilityMeasures.jl currently contains `Normal`, `LogNormal`, `Exponential`,
`Weibull`, `Gamma`, `Uniform`, `Cauchy`, `Laplace`, `Categorical`, `Bernoulli`,
`Binomial`, `Poisson`, `Geometric`, `MvNormal`, and `Multinomial`. Transformed or
composite measures and Distributions.jl interoperability are not implemented yet.

## Citation

If you use ProbabilityMeasures.jl in published work, citation metadata is available
in [CITATION.cff](CITATION.cff).
