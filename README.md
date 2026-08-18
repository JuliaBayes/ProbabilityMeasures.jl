# ProbabilityMeasures.jl

[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/dev)
[![Tests](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl)

ProbabilityMeasures.jl provides normalized probability measures designed for use in
probabilistic programs. Its implementations are type-generic, allocation-free in core
density and sampling operations, and compatible with automatic differentiation,
broadcasting on GPU arrays, and Reactant tracing.

The package is experimental. At present it implements the univariate normal,
exponential, and uniform measures, and the multivariate normal.

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

`Normal(μ, σ)`, `Exponential(θ)`, and `Uniform(a, b)` each support:

- `densityof` and `logdensityof`
- `cdf`, `ccdf`, `logcdf`, and `logccdf`
- `quantile`, `mean`, `median`, `var`, `std`, and `entropy`
- `rand` and Random's array-sampling methods
- `params`, `support`, `insupport`, and `checkparams`

`MvNormal(μ, L)` takes the lower-triangular Cholesky factor of the covariance, not the
covariance itself: a draw is then `μ + L z`, so sampling stays differentiable in the
parameters and the density costs one triangular solve rather than a factorization per
evaluation. Factor once at the call site if you hold a covariance.

```julia
using LinearAlgebra, ProbabilityMeasures

Σ = [4.0 1.0; 1.0 2.5]
d = MvNormal([1.0, -2.0], Matrix(cholesky(Σ).L))

logdensityof(d, [0.3, -1.0])
mean(d), cov(d), var(d), std(d), entropy(d)
rand(d)
```

It supports `densityof`, `logdensityof`, `rand`, `mean`, `cov`, `var`, `std`, `entropy`,
`params`, `support`, `insupport`, and `checkparams`. `var` and `std` are the marginals,
the diagonal of `cov` and its elementwise square root. There is no `cdf`, `quantile`, or
`median`: none of them has a closed form in more than one dimension. Its `logdensityof`
also allocates, where the univariate measures' do not.

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
respect to the parameters without custom derivative rules.

## Defining a measure

A subtype of `AbstractProbabilityMeasure` must implement:

- `DensityInterface.logdensityof(d, x)`
- `Base.rand(rng::AbstractRNG, d)`
- `Base.eltype(::Type{typeof(d)})`
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

ProbabilityMeasures.jl currently contains `Normal`, `Exponential`, `Uniform`, and
`MvNormal`. Discrete measures, transformed or composite measures, and Distributions.jl
interoperability are not implemented yet.

## Citation

If you use ProbabilityMeasures.jl in published work, citation metadata is available
in [CITATION.cff](CITATION.cff).
