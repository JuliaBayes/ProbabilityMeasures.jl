# Style guide

Rules for writing code in this repository. They are about *style*, not architecture:
how code reads, not how it is organized.

Before writing or editing comments and docstrings, read the surrounding file and
match its tone, structure, and level of detail. Use plain, direct language. Explain
surprising behavior, constraints, and choices that a reader could not infer from the
code. Do not restate obvious code, speculate about future work, or add background
that does not help someone use or maintain the package. Keep the shortest wording
that preserves the useful detail.

## Comments

**Multi-line comments use `#= =#`.** A stack of `#` lines is not idiomatic Julia.
Single-line comments use `#`.

```julia
#=
  A longer note that needs several lines to make its point.
=#

# A short note.
```

**Keep comments short and plainly worded.** A comment earns its place by saying
something the code does not already say. Restating the signature, spelling out an
indexing formula, or narrating each step is noise.

Too verbose:

```julia
# In-place SPD banded solver. Wraps LAPACK's `?pbsv` (not exposed in
# `LinearAlgebra.LAPACK`). Solves A·X = B where A is SPD with
# bandwidth `kd`, stored in upper-banded format (`uplo='U'`):
# A[i,j] = AB[kd+1+i-j, j] for max(1, j-kd) ≤ i ≤ j.
# Both `AB` (Cholesky factor on return) and `B` (solution on return)
# are overwritten. Falls back to the dense path for non-BLAS eltypes.
```

Better:

```julia
#=
  Solves A·X = B for banded SPD A via LAPACK's `?pbsv`, which
  `LinearAlgebra.LAPACK` does not expose. `AB` and `B` are overwritten with the
  Cholesky factor and the solution. Non-BLAS eltypes take the dense path.
=#
```

The storage-layout detail is not lost, it belongs in the docstring, where a caller
will look for it. A comment is for the reader of the implementation.

**No decorative separators.** These are visually intrusive:

```julia
# --- Moments --------------------------------------------------------------
# ==========================================================================
############################ Distribution functions
```

If a file needs sections, a blank line and a plain comment are enough. If it needs
them badly, it probably needs splitting.

**Write like a person, not like a model.** In particular, em dashes must be
grammatically necessary, not sprinkled in for rhythm. Most sentences that reach for
one want a comma, a colon, or a full stop instead. The same goes for other tics:
"Note that", "It's worth noting", "deliberately", "by design", and paired
constructions like "not X, but Y" used purely for cadence.

## Docstrings

Follow the conventions in the Julia manual unless there is a specific reason not to.
The shape:

```julia
"""
    fit_model(data, n_states; maxiter=100, tolerance=1e-6, rng=Random.default_rng())

Fit an `n_states`-state model to `data`.

# Arguments

- `data`: Observations used for fitting.
- `n_states::Integer`: Number of latent states.

# Keywords

- `maxiter::Integer=100`: Maximum number of optimization iterations.
- `tolerance::Real=1e-6`: Convergence threshold.
- `rng::AbstractRNG=Random.default_rng()`: Random-number generator.

# Returns

A [`FittedModel`](@ref) containing the estimated parameters and optimization
diagnostics.
"""
```

Points to keep:

- The signature comes first, indented four spaces, so it renders as code.
- Then a one-line summary in the imperative mood ("Fit an ...", not "Fits an ...").
- Section headers are `#`-level: `# Arguments`, `# Keywords`, `# Returns`, and
  `# Examples` where a doctest helps.
- Refer to other docstrings with `` [`name`](@ref) ``.
- Mark up identifiers with backticks and math with ` ```math ` blocks or ```` ``
  inline LaTeX.

Not every docstring needs every section. A two-argument function whose arguments are
obvious from the summary does not need `# Arguments`. Add sections when they carry
information, and skip them when they only pad.

## Formatting

Formatting is handled by JuliaFormatter, configured in `.JuliaFormatter.toml`: blue
style, four-space indent, 92-column margin. Run it rather than hand-aligning code.
`.editorconfig` covers whitespace and line endings.
