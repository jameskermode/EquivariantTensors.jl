#=
Reactant-Compatible ACE Evaluation
==================================

This module provides Reactant-compilable implementations of the ACE kernel,
bypassing KernelAbstractions which cannot be traced by Reactant.

Key changes from standard ACE:
1. Replace KA @kernel macros with plain Julia loops
2. Convert spec tuples to matrices for indexing
3. Convert SparseMatCSX to dense matrices
4. Use broadcasting where possible

Usage:
    using Reactant
    include("ace_reactant.jl")

    # Prepare state from standard ACE state
    rst = prepare_reactant_state(st)

    # Compile with Reactant
    f = Reactant.@compile ace_evaluate_reactant(Rnl_ra, Ylm_ra, rst...)
=#

import EquivariantTensors as ET

## ============================================================================
## Data Structure Conversion
## ============================================================================

"""
    spec_to_matrix(spec::Vector{<:Tuple})

Convert a vector of tuples (spec for PooledSparseProduct or SparseSymmProd)
to a matrix for Reactant-compatible indexing.

Input:  Vector of N-tuples, e.g., [(1,2), (3,4), (5,6)]
Output: Matrix of size (length(spec), N), e.g., [1 2; 3 4; 5 6]
"""
function spec_to_matrix(spec::Vector{<:Tuple})
    isempty(spec) && return zeros(Int, 0, 0)
    N = length(spec[1])
    mat = zeros(Int, length(spec), N)
    for (i, ϕ) in enumerate(spec)
        for j in 1:N
            mat[i, j] = ϕ[j]
        end
    end
    return mat
end

"""
    sparse_to_dense(m::ET.SparseMatCSX)

Convert a SparseMatCSX to a dense matrix.
For typical ACE coupling matrices, this is acceptable since they are small.
"""
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

## ============================================================================
## ReactantACEState - Pre-processed state for Reactant compilation
## ============================================================================

"""
    ReactantACEState{T}

Pre-processed ACE state with data structures converted to Reactant-compatible formats:
- Spec tuples converted to matrices
- Sparse matrices converted to dense
"""
struct ReactantACEState{T, VI <: AbstractVector{Int}, VMI <: AbstractVector{<:AbstractMatrix{Int}}, VMT <: AbstractVector{<:AbstractMatrix{T}}}
    spec_R::VI                    # Radial indices from aspec
    spec_Y::VI                    # Angular indices from aspec
    specs_mats::VMI               # Converted aaspecs (vector of matrices)
    A2Bmaps_dense::VMT            # Dense coupling matrices
end

"""
    prepare_reactant_state(st)

Convert standard ACE state to ReactantACEState for Reactant compilation.
"""
function prepare_reactant_state(st)
    # Extract spec indices from tuple format
    spec_R = [s[1] for s in st.aspec]
    spec_Y = [s[2] for s in st.aspec]

    # Convert aaspecs tuples to matrices
    specs_mats = [spec_to_matrix(Vector(s)) for s in st.aaspecs]

    # Convert sparse to dense
    A2Bmaps_dense = [sparse_to_dense(m) for m in st.A2Bmaps]

    return ReactantACEState(spec_R, spec_Y, specs_mats, A2Bmaps_dense)
end

"""
    unpack_reactant_state(rst::ReactantACEState)

Unpack ReactantACEState into individual components for function arguments.
"""
function unpack_reactant_state(rst::ReactantACEState)
    return (rst.spec_R, rst.spec_Y, rst.specs_mats, rst.A2Bmaps_dense)
end

## ============================================================================
## PooledSparseProduct - Reactant-compatible implementation
## ============================================================================

"""
    pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)

Reactant-compatible pooled sparse product using gather + vectorized sum.

Computes: A[inode, iA] = Σⱼ Rnl[j, inode, spec_R[iA]] * Ylm[j, inode, spec_Y[iA]]

Replaces the KA kernel `_ka_evaluate_PooledSparseProduct_batched_v1!`

Uses vectorized gather operations instead of scalar indexing for Reactant compatibility.
"""
function pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)
    maxneigs, nnodes, nRnl = size(Rnl_3)
    nA = length(spec_R)
    T = eltype(Rnl_3)

    # Gather: Rnl_gathered[j, inode, iA] = Rnl_3[j, inode, spec_R[iA]]
    # Using advanced indexing: Rnl_3[:, :, spec_R] gives [maxneigs, nnodes, nA]
    Rnl_gathered = Rnl_3[:, :, spec_R]
    Ylm_gathered = Ylm_3[:, :, spec_Y]

    # Elementwise product and sum over neighbors
    # prod[j, inode, iA] = Rnl_gathered[j, inode, iA] * Ylm_gathered[j, inode, iA]
    # A[inode, iA] = sum(prod[:, inode, iA])
    prod = Rnl_gathered .* Ylm_gathered

    # Sum over first dimension (neighbors), then transpose
    A = dropdims(sum(prod, dims=1), dims=1)  # [nnodes, nA]

    return A
end

## ============================================================================
## SparseSymmProd - Reactant-compatible implementation
## ============================================================================

"""
    sparse_symm_prod_order(A, spec_mat, order)

Compute symmetric product for a single correlation order using vectorized gather.

For order=1: AA[inode, i] = A[inode, spec_mat[i, 1]]
For order=2: AA[inode, i] = A[inode, spec_mat[i, 1]] * A[inode, spec_mat[i, 2]]
etc.
"""
function sparse_symm_prod_order(A, spec_mat, order::Int)
    nnodes = size(A, 1)
    nspec = size(spec_mat, 1)

    if order == 0
        # Order 0: just ones
        return ones(eltype(A), nnodes, nspec)
    end

    # Start with product = 1
    # For each t in 1:order, gather A[:, spec_mat[:, t]] and multiply
    # spec_mat[:, t] gives indices for term t across all specs

    # First term: A[:, spec_mat[:, 1]] gives [nnodes, nspec]
    prod = A[:, spec_mat[:, 1]]

    # Multiply by remaining terms
    for t in 2:order
        prod = prod .* A[:, spec_mat[:, t]]
    end

    return prod
end

"""
    sparse_symm_prod_reactant(A, specs_mats)

Reactant-compatible sparse symmetric product using vectorized gather.

Computes: AA[inode, offset+i] = Πₜ A[inode, specs_mats[ord][i, t]]

specs_mats is a vector of matrices, where specs_mats[ord] has shape (nspec, ord)
containing the indices for products of that order.

Replaces the KA kernel `_ka_evaluate_SparseSymmProd_batched_v1!`
"""
function sparse_symm_prod_reactant(A, specs_mats)
    nnodes = size(A, 1)

    # Process each order and concatenate
    AA_parts = [sparse_symm_prod_order(A, spec_mat, size(spec_mat, 2)) for spec_mat in specs_mats]

    # Concatenate along second dimension
    AA = hcat(AA_parts...)

    return AA
end

## ============================================================================
## Full ACE Evaluation - Reactant-compatible
## ============================================================================

"""
    ace_evaluate_reactant(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmaps_dense)

Full Reactant-compatible ACE evaluation.

Input:
- Rnl_3: (maxneigs, nnodes, nRnl) radial embeddings
- Ylm_3: (maxneigs, nnodes, nYlm) angular embeddings
- spec_R, spec_Y: indices for pooled product
- specs_mats: vector of index matrices for symmetric products
- A2Bmaps_dense: dense coupling matrices

Output:
- BB: tuple of (nnodes, nfeatures) basis arrays
- A: (nnodes, nA) intermediate A array
- AA: (nnodes, nAA) intermediate AA array
"""
function ace_evaluate_reactant(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmaps_dense)
    # Step 1: Pooled sparse product
    # A[inode, iA] = Σⱼ Rnl[j, inode, ϕR] * Ylm[j, inode, ϕY]
    A = pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)

    # Step 2: Sparse symmetric product
    # AA[inode, i] = Πₜ A[inode, ϕ[t]]
    AA = sparse_symm_prod_reactant(A, specs_mats)

    # Step 3: Apply coupling coefficients (dense matmul)
    # BB[i] = permutedims(A2Bmaps[i] * permutedims(AA))
    # Which is equivalent to: BB[i] = AA * A2Bmaps[i]'
    BB = Tuple(permutedims(m * permutedims(AA)) for m in A2Bmaps_dense)

    return BB, A, AA
end

"""
    ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)

Simplified version for single output (L=0 scalar invariants).
"""
function ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    A = pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)
    AA = sparse_symm_prod_reactant(A, specs_mats)
    BB = permutedims(A2Bmap * permutedims(AA))
    return BB, A, AA
end

## ============================================================================
## Energy Function - Ready for Reactant compilation
## ============================================================================

"""
    ace_energy_reactant(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)

Compute ACE energy for Reactant compilation.

Returns total energy as a scalar.
"""
function ace_energy_reactant(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
    BB, _, _ = ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    return sum(BB * params)
end

## ============================================================================
## Utility: Convert standard ACE evaluation inputs/outputs
## ============================================================================

"""
    verify_reactant_vs_ka(symbasis, Rnl_3, Ylm_3, st; rtol=1e-5)

Verify that Reactant-compatible evaluation matches KA version.
Returns (max_error_A, max_error_AA, max_error_BB)
"""
function verify_reactant_vs_ka(symbasis, Rnl_3, Ylm_3, st; rtol=1e-5)
    # Standard KA evaluation
    BB_ka, A_ka, AA_ka = ET._ka_evaluate(symbasis, Rnl_3, Ylm_3,
                                          st.aspec, st.aaspecs, st.A2Bmaps)

    # Reactant-compatible evaluation
    rst = prepare_reactant_state(st)
    BB_rt, A_rt, AA_rt = ace_evaluate_reactant(Rnl_3, Ylm_3,
                                                rst.spec_R, rst.spec_Y,
                                                rst.specs_mats, rst.A2Bmaps_dense)

    err_A = maximum(abs.(A_rt .- A_ka))
    err_AA = maximum(abs.(AA_rt .- AA_ka))
    err_BB = maximum(maximum(abs.(BB_rt[i] .- BB_ka[i])) for i in 1:length(BB_ka))

    return (A=err_A, AA=err_AA, BB=err_BB)
end

## ============================================================================
## Embedding Functions (Self-Contained)
## ============================================================================

# Helper functions for spherical harmonic indexing
_sizeY(maxL) = (maxL + 1)^2
_lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
    evaluate_reactant_cheb(N, x)

Reactant-compatible Chebyshev polynomial evaluation using broadcasting.
"""
function evaluate_reactant_cheb(N::Int, x::AbstractVector{T}) where {T}
    P = similar(x, T, length(x), N)
    P[:, 1] .= one(T)
    if N > 1; P[:, 2] .= x; end
    for k = 3:N
        @views P[:, k] .= 2 .* x .* P[:, k-1] .- P[:, k-2]
    end
    return P
end

"""
    compute_radial_embedding(x, y, z, N_cheb, rcut)

Compute radial embeddings using Reactant-compatible operations.
"""
function compute_radial_embedding(x, y, z, N_cheb::Int, rcut)
    T = eltype(x)

    # Radial transform: y = 1 / (1 + r^2)
    r_sq = x .* x .+ y .* y .+ z .* z
    y_trans = one(T) ./ (one(T) .+ r_sq)

    # Envelope: (y - ycut)^2 * (y + ycut)^2
    ycut = one(T) / (one(T) + T(rcut)^2)
    env = (y_trans .- ycut).^2 .* (y_trans .+ ycut).^2

    # Evaluate Chebyshev basis and apply envelope
    return evaluate_reactant_cheb(N_cheb, y_trans) .* env
end

"""
    compute_reactant_ylm(L, Flm, x, y, z)

Reactant-compatible solid harmonics evaluation using broadcasting.
"""
function compute_reactant_ylm(L::Int, Flm, x::AbstractVector{T},
                               y::AbstractVector{T}, z::AbstractVector{T}) where {T}
    nX = length(x)
    len = _sizeY(L)
    Z = similar(x, T, nX, len)
    rt2 = sqrt(T(2))

    r² = x .* x .+ y .* y .+ z .* z
    s = similar(Z, nX, L+1)
    c = similar(Z, nX, L+1)
    s[:, 1] .= zero(T); c[:, 1] .= one(T)
    for m = 1:L
        @views s[:, m+1] .= s[:, m] .* x .+ c[:, m] .* y
        @views c[:, m+1] .= c[:, m] .* x .- s[:, m] .* y
    end
    Q = similar(Z, nX, len)
    i00 = _lm2idx(0, 0)
    Q[:, i00] .= one(T)
    Z[:, i00] .= (Flm[1,1]/rt2) .* Q[:, i00]
    c[:, 1] .= one(T)/rt2

    for l = 1:L
        ill = _lm2idx(l, l); il⁻l = _lm2idx(l, -l)
        ill⁻¹ = _lm2idx(l, l-1); il⁻¹l⁻¹ = _lm2idx(l-1, l-1)
        il⁻l⁺¹ = _lm2idx(l, -l+1)
        F_l_l = Flm[1+l,1+l]; F_l_l⁻¹ = Flm[1+l,1+l-1]
        @views Q[:, ill] .= -(2*l-1) .* Q[:, il⁻¹l⁻¹]
        @views Z[:, ill] .= F_l_l .* Q[:, ill] .* c[:, l+1]
        @views Z[:, il⁻l] .= F_l_l .* Q[:, ill] .* s[:, l+1]
        @views Q[:, ill⁻¹] .= (2*l-1) .* z .* Q[:, il⁻¹l⁻¹]
        @views Z[:, il⁻l⁺¹] .= F_l_l⁻¹ .* Q[:, ill⁻¹] .* s[:, l]
        @views Z[:, ill⁻¹] .= F_l_l⁻¹ .* Q[:, ill⁻¹] .* c[:, l]
        for m = l-2:-1:0
            ilm = _lm2idx(l, m); il⁻m = _lm2idx(l, -m)
            il⁻¹m = _lm2idx(l-1, m); il⁻²m = _lm2idx(l-2, m)
            F_l_m = Flm[1+l,1+m]
            @views Q[:, ilm] .= ((2*l-1) .* z .* Q[:, il⁻¹m] .- (l+m-1) .* r² .* Q[:, il⁻²m]) ./ (l-m)
            @views Z[:, il⁻m] .= F_l_m .* Q[:, ilm] .* s[:, m+1]
            @views Z[:, ilm] .= F_l_m .* Q[:, ilm] .* c[:, m+1]
        end
    end
    return Z
end

## ============================================================================
## Full MLIP Energy with Embeddings
## ============================================================================

"""
    mlip_energy_reactant(edge_x, edge_y, edge_z, first, maxneigs, nnodes,
                         N_cheb, maxl, rcut, Flm,
                         spec_R, spec_Y, specs_mats, A2Bmap, params)

Full MLIP energy evaluation ready for Reactant compilation.

Includes:
1. Radial embedding (Chebyshev + envelope)
2. Angular embedding (solid harmonics)
3. Reshape to 3D
4. ACE evaluation
5. Linear readout
"""
function mlip_energy_reactant(edge_x, edge_y, edge_z, first, maxneigs, nnodes,
                               N_cheb, maxl, rcut, Flm,
                               spec_R, spec_Y, specs_mats, A2Bmap, params)
    # Compute embeddings
    Rnl = compute_radial_embedding(edge_x, edge_y, edge_z, N_cheb, rcut)
    Ylm = compute_reactant_ylm(maxl, Flm, edge_x, edge_y, edge_z)

    # Reshape to 3D arrays
    Rnl_3 = reshape_to_3d(Rnl, first, maxneigs, nnodes)
    Ylm_3 = reshape_to_3d(Ylm, first, maxneigs, nnodes)

    # ACE evaluation
    BB, _, _ = ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)

    # Readout
    return sum(BB * params)
end

"""
    reshape_to_3d(X, first, maxneigs, nnodes)

Reshape a (nedges, nfeatures) array to (maxneigs, nnodes, nfeatures)
using neighbour list structure.

`first[i]` is the index of the first edge for node i.
"""
function reshape_to_3d(X::AbstractMatrix, first::AbstractVector{Int}, maxneigs::Int, nnodes::Int)
    nedges, nfeatures = size(X)
    T = eltype(X)

    X_3d = zeros(T, maxneigs, nnodes, nfeatures)

    for inode in 1:nnodes
        start_idx = first[inode]
        end_idx = inode < nnodes ? first[inode+1] - 1 : nedges
        nneig = min(end_idx - start_idx + 1, maxneigs)

        for j in 1:nneig
            edge_idx = start_idx + j - 1
            for k in 1:nfeatures
                X_3d[j, inode, k] = X[edge_idx, k]
            end
        end
    end

    return X_3d
end
