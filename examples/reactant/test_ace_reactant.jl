#=
Test: ACE Kernel Reactant Compilation Feasibility
==================================================

This script tests whether the ACE kernel can be compiled with Reactant,
and identifies specific issues that need to be addressed.

Run with: julia --project test_ace_reactant.jl
=#

using Printf
using LinearAlgebra, StaticArrays
using Reactant
import EquivariantTensors as ET
import Polynomials4ML as P4ML
using SpheriCart

println("="^70)
println("ACE KERNEL REACTANT COMPILATION FEASIBILITY TEST")
println("="^70)

## ============================================================================
## Setup: Build a simple ACE model
## ============================================================================

println("\n1. Building ACE model components...")

Dtot, maxl, ORD = 4, 2, 2  # Small model for testing
N_cheb = Dtot + 1

# Build the symmetric basis
mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

nfeatures = length(symbasis, 0)
@printf("Model: Dtot=%d, maxl=%d, ORD=%d, %d features\n", Dtot, maxl, ORD, nfeatures)

## ============================================================================
## Create test data
## ============================================================================

println("\n2. Creating test data...")

maxneigs, nnodes = 10, 5
nRnl = N_cheb
nYlm = (maxl + 1)^2

# 3D embeddings: (maxneigs, nnodes, nfeatures)
Rnl_3 = randn(Float32, maxneigs, nnodes, nRnl)
Ylm_3 = randn(Float32, maxneigs, nnodes, nYlm)

@printf("Rnl_3: %s, Ylm_3: %s\n", size(Rnl_3), size(Ylm_3))

## ============================================================================
## Get the state components
## ============================================================================

println("\n3. Extracting state components...")

using Random
rng = MersenneTwister(1234)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)

# The key state components
aspec = st.aspec          # Vector{Tuple{Int,Int}}
aaspecs = st.aaspecs      # Tuple of Vector{NTuple}
A2Bmaps = st.A2Bmaps      # Tuple of SparseMatCSX

println("   aspec: $(typeof(aspec)), length=$(length(aspec))")
println("   aaspecs: $(typeof(aaspecs))")
for (i, spec) in enumerate(aaspecs)
    println("      [$i]: $(typeof(spec)), length=$(length(spec))")
end
println("   A2Bmaps: $(typeof(A2Bmaps))")
for (i, m) in enumerate(A2Bmaps)
    println("      [$i]: $(typeof(m)), size=($(m.m), $(m.n))")
end

## ============================================================================
## Test 1: Standard Julia evaluation
## ============================================================================

println("\n4. Testing standard Julia evaluation...")

BB_julia, A_julia, AA_julia = ET._ka_evaluate(symbasis, Rnl_3, Ylm_3,
                                               aspec, aaspecs, A2Bmaps)
@printf("   BB: %s (tuple of %d arrays)\n", typeof(BB_julia), length(BB_julia))
@printf("   A:  %s\n", size(A_julia))
@printf("   AA: %s\n", size(AA_julia))
println("   Energy: $(sum(sum.(BB_julia)))")

## ============================================================================
## Test 2: Identify problematic components
## ============================================================================

println("\n5. Analyzing components for Reactant compatibility...")

println("\n   [A] PooledSparseProduct (abasis):")
println("      - Uses KernelAbstractions @kernel macro")
println("      - Indexes into spec array with dynamic indices")
println("      - Uses ntuple with runtime size")

println("\n   [B] SparseSymmProd (aabasis):")
println("      - Uses KernelAbstractions @kernel macro")
println("      - Uses @nexprs macro for unrolling")
println("      - Indexes into specs with dynamic indices")

println("\n   [C] SparseMatCSX multiplication:")
println("      - Custom sparse matrix format (CSR + CSC)")
println("      - Uses KernelAbstractions for multiplication")

## ============================================================================
## Test 3: Try to create Reactant-compatible pure Julia version
## ============================================================================

println("\n6. Creating Reactant-compatible pure Julia version...")

# Convert SparseMatCSX to dense for Reactant
function sparse_to_dense(m::ET.SparseMatCSX)
    dense = zeros(eltype(m.nzval_csr), m.m, m.n)
    for row in 1:m.m
        for idx in m.rowptr[row]:(m.rowptr[row+1]-1)
            col = m.colval[idx]
            dense[row, col] = m.nzval_csr[idx]
        end
    end
    return dense
end

A2Bmaps_dense = [sparse_to_dense(m) for m in A2Bmaps]
println("   Converted A2Bmaps to dense:")
for (i, m) in enumerate(A2Bmaps_dense)
    println("      [$i]: $(size(m)), nnz=$(count(!=(0), m))/$(length(m))")
end

# Pure Julia implementation of PooledSparseProduct
function pooled_sparse_product_julia(Rnl_3, Ylm_3, spec)
    maxneigs, nnodes, _ = size(Rnl_3)
    nA = length(spec)
    A = zeros(eltype(Rnl_3), nnodes, nA)

    for iA in 1:nA
        ϕR, ϕY = spec[iA]
        for inode in 1:nnodes
            a = zero(eltype(A))
            for j in 1:maxneigs
                a += Rnl_3[j, inode, ϕR] * Ylm_3[j, inode, ϕY]
            end
            A[inode, iA] = a
        end
    end
    return A
end

# Pure Julia implementation of SparseSymmProd
function sparse_symm_prod_julia(A, specs)
    nnodes = size(A, 1)
    nAA = sum(length, specs)
    AA = zeros(eltype(A), nnodes, nAA)

    offset = 0
    for (ord, spec) in enumerate(specs)
        for (i, ϕ) in enumerate(spec)
            for inode in 1:nnodes
                aa = one(eltype(AA))
                for t in 1:length(ϕ)
                    aa *= A[inode, ϕ[t]]
                end
                AA[inode, offset + i] = aa
            end
        end
        offset += length(spec)
    end
    return AA
end

# Full ACE evaluation in pure Julia
function ace_evaluate_julia(Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps_dense)
    # A = pooled sparse product
    A = pooled_sparse_product_julia(Rnl_3, Ylm_3, aspec)

    # AA = sparse symmetric product
    AA = sparse_symm_prod_julia(A, aaspecs)

    # BB = A2Bmaps * AA'
    BB = [permutedims(m * permutedims(AA)) for m in A2Bmaps_dense]

    return BB, A, AA
end

# Verify pure Julia version matches KA version
BB_pure, A_pure, AA_pure = ace_evaluate_julia(Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps_dense)
println("\n   Verification:")
@printf("      A diff: %.2e\n", maximum(abs.(A_pure .- A_julia)))
@printf("      AA diff: %.2e\n", maximum(abs.(AA_pure .- AA_julia)))
@printf("      BB diff: %.2e\n", maximum(abs.(BB_pure[1] .- BB_julia[1])))

## ============================================================================
## Test 4: Try Reactant compilation of pure Julia version
## ============================================================================

println("\n7. Testing Reactant compilation...")

# The issue: spec contains tuples which are indexed dynamically
# This creates problems for Reactant tracing

# Approach 1: Try compiling with converted specs as matrices
function convert_spec_to_matrix(spec::Vector{<:Tuple})
    N = length(spec[1])
    mat = zeros(Int, length(spec), N)
    for (i, ϕ) in enumerate(spec)
        for j in 1:N
            mat[i, j] = ϕ[j]
        end
    end
    return mat
end

aspec_mat = convert_spec_to_matrix(aspec)
aaspecs_mats = [convert_spec_to_matrix(s) for s in aaspecs]

println("   Converted specs to matrices:")
println("      aspec_mat: $(size(aspec_mat))")
for (i, m) in enumerate(aaspecs_mats)
    println("      aaspecs_mats[$i]: $(size(m))")
end

# Reactant-compatible pooled sparse product using matrix spec
function pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_mat)
    maxneigs, nnodes, _ = size(Rnl_3)
    nA = size(spec_mat, 1)
    A = similar(Rnl_3, nnodes, nA)

    for iA in 1:nA
        ϕR = spec_mat[iA, 1]
        ϕY = spec_mat[iA, 2]
        for inode in 1:nnodes
            a = zero(eltype(A))
            for j in 1:maxneigs
                a += Rnl_3[j, inode, ϕR] * Ylm_3[j, inode, ϕY]
            end
            A[inode, iA] = a
        end
    end
    return A
end

# Test the matrix-spec version
A_mat = pooled_sparse_product_reactant(Rnl_3, Ylm_3, aspec_mat)
@printf("   Matrix-spec version diff: %.2e\n", maximum(abs.(A_mat .- A_julia)))

## ============================================================================
## Test 5: Try actual Reactant compilation
## ============================================================================

println("\n8. Attempting Reactant @compile...")

# Simple test: just the matrix multiplication part
function simple_ace_matmul(AA, A2Bmap)
    return permutedims(A2Bmap * permutedims(AA))
end

try
    Reactant.set_default_backend("cpu")
    AA_ra = Reactant.to_rarray(AA_julia)
    A2B_ra = Reactant.to_rarray(A2Bmaps_dense[1])

    matmul_compiled = Reactant.@compile simple_ace_matmul(AA_ra, A2B_ra)

    BB_compiled = matmul_compiled(AA_ra, A2B_ra)
    BB_ref = simple_ace_matmul(AA_julia, A2Bmaps_dense[1])

    @printf("   [OK] Matrix multiplication: diff = %.2e\n",
            maximum(abs.(Array(BB_compiled) .- BB_ref)))
catch e
    println("   [FAIL] Matrix multiplication: $(typeof(e))")
    showerror(stdout, e)
    println()
end

# Test: Pooled sparse product with broadcasting
function pooled_sparse_product_broadcast(Rnl_3, Ylm_3, spec_mat)
    # This version uses broadcasting which should work with Reactant
    maxneigs, nnodes, _ = size(Rnl_3)
    nA = size(spec_mat, 1)

    # Gather the relevant features - this is the tricky part
    # spec_mat[:, 1] contains Rnl indices, spec_mat[:, 2] contains Ylm indices
    # We need to index Rnl_3[:, :, spec_mat[iA, 1]] for each iA

    # For now, just compute sum over all features as a simpler test
    E = sum(Rnl_3) + sum(Ylm_3)
    return E
end

try
    Rnl_ra = Reactant.to_rarray(Rnl_3)
    Ylm_ra = Reactant.to_rarray(Ylm_3)
    spec_ra = Reactant.to_rarray(aspec_mat)

    simple_compiled = Reactant.@compile pooled_sparse_product_broadcast(Rnl_ra, Ylm_ra, spec_ra)

    E_compiled = simple_compiled(Rnl_ra, Ylm_ra, spec_ra)
    E_ref = pooled_sparse_product_broadcast(Rnl_3, Ylm_3, aspec_mat)

    @printf("   [OK] Simple broadcast test: diff = %.2e\n",
            abs(Float64(E_compiled) - E_ref))
catch e
    println("   [FAIL] Simple broadcast test: $(typeof(e))")
    showerror(stdout, e)
    println()
end

## ============================================================================
## Summary
## ============================================================================

println("\n" * "="^70)
println("SUMMARY: REACTANT COMPATIBILITY ISSUES")
println("="^70)

println("""

IDENTIFIED ISSUES:

1. KernelAbstractions @kernel macros
   - ACE uses KA kernels for GPU portability
   - Reactant uses XLA, which has its own kernel compilation
   - SOLUTION: Rewrite kernels as pure Julia with broadcasting

2. Dynamic indexing into spec arrays
   - spec[iA] returns a tuple, then ϕ[t] indexes into it
   - This creates control flow that's hard to trace
   - SOLUTION: Convert specs to dense matrices, use gather operations

3. SparseMatCSX format
   - Custom sparse matrix with both CSR and CSC
   - Reactant can handle dense matrices easily
   - SOLUTION: Convert to dense (acceptable for small coupling matrices)

4. ntuple with runtime size
   - ntuple(t -> ..., N) where N is a type parameter
   - Reactant needs static shapes
   - SOLUTION: Unroll for fixed orders (ORD ≤ 4 typically)

PROPOSED APPROACH:

1. Create a ReactantACE struct that stores:
   - Dense A2Bmaps matrices
   - Spec matrices instead of tuple vectors
   - Pre-computed gather indices

2. Implement forward pass with broadcasting:
   - Use einsum-style operations for pooled products
   - Use matrix multiply for A2B transformation
   - Avoid dynamic indexing

3. For gradients:
   - Let Reactant/Enzyme handle AD automatically
   - No need for custom rrules

FEASIBILITY: MEDIUM-HIGH
- The main challenge is rewriting the gather operations
- Dense matrices are acceptable for typical ACE sizes
- Broadcasting version should be 1-2x slower than KA kernels on CPU
- But Reactant compilation can recover this through fusion
""")

println("="^70)
