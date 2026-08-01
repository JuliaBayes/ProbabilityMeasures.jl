# ProbabilityMeasures

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/dev)
[![Test workflow status](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl)
[![Docs workflow Status](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/rsenne/ProbabilityMeasures.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

A library of normalized probability measures built for probabilistic programming:
type-generic, allocation-free, and clean under automatic differentiation and on the
GPU.

## Why not Distributions.jl

Three things make Distributions.jl an awkward foundation for a PPL, and each one is
a design decision here rather than a patch:

**Parameters are not promoted.** `Distributions.Normal(μ, σ)` calls `promote`, so an
AD dual entering through `μ` widens `σ` as well, and a `Float32` GPU parameter
silently becomes `Float64` the moment it meets a literal. Here each parameter keeps
its own type:

```julia
julia> typeof(Normal(ForwardDiff.Dual(0.0, 1.0), 1.0))
Normal{ForwardDiff.Dual{Nothing, Float64, 1}, Float64}   # σ untouched

julia> logdensityof(Normal(0, 1), 1.0f0) isa Float32
true                                                     # precision follows x
```

**`logdensityof` is total.** It never throws -- correctly-typed `-Inf` outside the
support, `NaN` for invalid parameters. A function that can throw cannot be called
from inside a GPU kernel, and a PPL hands measures invalid parameters constantly
during warmup, line search, and rejected proposals.

**Constructors never validate.** `Normal(0.0, -1.0)` builds without complaint;
`checkparams(d)` is the opt-in check, for the boundaries where a human supplied the
numbers rather than the inner loop of a sampler.

## Batching and the GPU

Measures hold scalar, `isbits` parameters, so broadcasting is the batching
mechanism and it fuses into a single kernel with no wrapper type and no shape
algebra:

```julia
d  = Normal(0.0f0, 1.0f0)
xs = CUDA.randn(Float32, 10^6)
logdensityof.(d, xs)                      # one kernel

mus = CUDA.randn(Float32, 10^6)
logdensityof.(Normal.(mus, 1.0f0), xs)    # still one kernel
```

## The conformance suite

`libs/ProbabilityMeasuresTest` holds a reusable suite that every measure must pass.
It is what makes the claims above enforced rather than merely asserted:

| Property | Checked with |
| --- | --- |
| Interface conformance | Interfaces.jl |
| Correctness | density integrates to 1 (QuadGK), cdf↔quantile round-trip, Monte Carlo moments |
| Type genericity | `Float32`/`Float64`/`BigFloat` sweep, mixed and exact parameter types |
| Type stability | `@inferred` and JET |
| Allocations | AllocCheck, statically over the whole call graph |
| AD | ForwardDiff, ReverseDiff, Zygote and Mooncake against FiniteDifferences |
| GPU | JLArrays, so scalar-indexing and non-`isbits` capture are caught in CI with no device |

```julia
using ProbabilityMeasuresTest
test_measure(Normal(0.0, 1.0))
```

Keeping it in `libs/` rather than in the package is deliberate: JET, AllocCheck,
four AD backends, JLArrays and QuadGK stay out of the dependency graph, so a PPL
that depends on this package stays cheap to load.

## Scope

The exported surface is kept deliberately small — every name is one a PPL is
expected to call. Adding an export later is a non-breaking change and removing one
is not, so anything speculative costs more to ship now than to withhold.

Absent on purpose, and cheap to add when something needs them: `mode`, `skewness`,
`kurtosis`, `mgf`, `cf` (Distributions.jl inheritance rather than inference);
`Matrixvariate`; the `PositiveReals`/`UnitInterval`/`RealInterval` supports, which
will arrive with the first measure that has one.

## Status

Early. `Normal` is implemented and passes the conformance suite. Product and power
measures, transforms, and a Distributions.jl interop extension are next.

## How to Cite

If you use ProbabilityMeasures.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/rsenne/ProbabilityMeasures.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://rsenne.github.io/ProbabilityMeasures.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
