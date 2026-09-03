#=
  Triangular linear algebra for the measures parameterized by a Cholesky factor. Each
  routine builds new arrays instead of writing into one, so that reverse-mode
  differentiation tools, which reject array mutation, can follow it.
=#

"""
    rowdot(L, v, i, k)

The inner product of the first `k` entries of row `i` of `L` and `v`.
"""
@inline function rowdot(L::AbstractMatrix, v::AbstractVector, i::Integer, k::Integer)
    # Zero times infinity is `NaN`, which must remain visible in the result.
    acc = L[i, 1] * v[1]
    for j in 2:k
        acc = muladd(L[i, j], v[j], acc)
    end
    return acc
end

"""
    rowsdot(A, i, j, k)

The inner product of the first `k` entries of rows `i` and `j` of `A`.
"""
@inline function rowsdot(A::AbstractMatrix, i::Integer, j::Integer, k::Integer)
    acc = A[i, 1] * A[j, 1]
    for m in 2:k
        acc = muladd(A[i, m], A[j, m], acc)
    end
    return acc
end

"""
    forwardsolve(L, b)

``L^{-1} b`` for lower-triangular `L`, by forward substitution.

Only the lower triangle of `L` is read.
"""
function forwardsolve(L::AbstractMatrix{<:Number}, b::AbstractVector{<:Number})
    # The first entry sets the result type and keeps later inner products non-empty.
    z = [b[1] / L[1, 1]]
    for i in 2:length(b)
        z = vcat(z, (b[i] - rowdot(L, z, i, i - 1)) / L[i, i])
    end
    return z
end

"""
    logdetdiag(A, n, ::Type{T})

``\\sum_{i=1}^{n} \\log A_{ii}`` in `float(T)`.

A non-positive diagonal entry gives a non-finite result instead of an error, which is
what makes the log-determinant of a triangular factor safe to evaluate on unvalidated
parameters.
"""
function logdetdiag(A, n::Integer, ::Type{T}) where {T}
    R = float(T)
    acc = zero(R)
    for i in 1:n
        acc += logt(convert(R, A[i, i]))
    end
    return acc
end

"""
    cholfactor(X)

The lower-triangular Cholesky factor `C` of `X`, with `C * C' == X`.

Only the lower triangle of `X` is read. Where `X` is not positive definite, the pivot
goes through [`sqrtt`](@ref) and the factor carries `NaN` from that column on, so a
caller gets a non-finite result rather than an error.
"""
function cholfactor(X::AbstractMatrix{<:Number})
    n = size(X, 1)
    pivot = sqrtt(X[1, 1])
    C = reshape(map(i -> X[i, 1] / pivot, 1:n), n, 1)
    for j in 2:n
        C = hcat(C, cholcolumn(X, C, j, n))
    end
    return C
end

#=
  Column `j` of the factor, given the `j - 1` columns before it: zero above the diagonal,
  the pivot on it, and the rest of column `j` of `X` with those columns subtracted off.

  This is a function rather than a loop body because the pivot is captured by the closure
  that fills the column. A captured variable that the enclosing scope reassigns is boxed,
  and its contents are then opaque, which widens the whole factor to `Any`.
=#
function cholcolumn(X::AbstractMatrix, C::AbstractMatrix, j::Integer, n::Integer)
    pivot = sqrtt(X[j, j] - rowsdot(C, j, j, j - 1))
    return map(i -> i < j ? zero(pivot) : (X[i, j] - rowsdot(C, i, j, j - 1)) / pivot, 1:n)
end
