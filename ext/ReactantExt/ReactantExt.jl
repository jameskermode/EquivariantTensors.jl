"""
ReactantExt: Extension for Reactant.jl compatibility

This extension provides:
1. `_is_reactant_traced()` detection for all Reactant array types
2. Pure Julia evaluation paths that avoid KernelAbstractions for Reactant tracing
3. Separate integer spec arrays instead of Vector{Tuple} for MLIR compatibility
"""
module ReactantExt

using Reactant: TracedRArray, TracedRNumber, AbstractConcreteArray
import EquivariantTensors

# ========================================================================
# Reactant detection - add methods for Reactant types
# The fallback ::Any method is defined in main EquivariantTensors module
#
# We detect all Reactant array types to route to pure Julia paths:
# - TracedRArray: Used during @compile tracing (elements are TracedRNumber)
# - TracedRNumber: Traced scalar values
# - AbstractConcreteArray: Concrete data arrays (from to_rarray, includes ConcretePJRTArray)
# ========================================================================

EquivariantTensors._is_reactant_traced(::TracedRArray) = true
EquivariantTensors._is_reactant_traced(::TracedRNumber) = true
EquivariantTensors._is_reactant_traced(::AbstractConcreteArray) = true
EquivariantTensors._is_reactant_traced(t::Tuple) = any(EquivariantTensors._is_reactant_traced, t)
# Handle wrapped arrays (ReshapedArray, SubArray, etc.)
EquivariantTensors._is_reactant_traced(a::Base.ReshapedArray) = EquivariantTensors._is_reactant_traced(parent(a))
EquivariantTensors._is_reactant_traced(a::SubArray) = EquivariantTensors._is_reactant_traced(parent(a))

# ========================================================================
# Pure Julia evaluation for PooledSparseProduct (NB=2 case)
#
# For Reactant compatibility, we use separate spec_R, spec_Y arrays
# instead of Vector{NTuple{2, Int}}. This avoids tuple indexing
# which cannot be represented in MLIR.
# ========================================================================

import EquivariantTensors: PooledSparseProduct, TupTen3

"""
    reactant_evaluate_pooled(Rnl_3, Ylm_3, spec_R, spec_Y)

Reactant-compatible pooled sparse product for NB=2.

Arguments:
- Rnl_3: (maxneigs, nnodes, nRnl) - radial embeddings
- Ylm_3: (maxneigs, nnodes, nYlm) - angular embeddings
- spec_R: Vector{Int} - radial indices for each basis function
- spec_Y: Vector{Int} - angular indices for each basis function

Returns:
- A: (nnodes, nA) - pooled basis values
"""
function reactant_evaluate_pooled(Rnl_3::AbstractArray{T,3},
                                   Ylm_3::AbstractArray{T,3},
                                   spec_R::AbstractVector{<:Integer},
                                   spec_Y::AbstractVector{<:Integer}) where {T}
    maxneigs, nnodes, _ = size(Rnl_3)
    nA = length(spec_R)

    # Vectorized gather: (maxneigs, nnodes, nA)
    Rnl_gathered = Rnl_3[:, :, spec_R]
    Ylm_gathered = Ylm_3[:, :, spec_Y]

    # Element-wise product
    prod_RY = Rnl_gathered .* Ylm_gathered

    # Sum over neighbors: (maxneigs, nnodes, nA) -> (nnodes, nA)
    A = dropdims(sum(prod_RY, dims=1), dims=1)

    return A
end

"""
    reactant_pullback_pooled(∂A, Rnl_3, Ylm_3, spec_R, spec_Y)

Reactant-compatible pullback for pooled sparse product.

Returns gradients w.r.t. Rnl_3 and Ylm_3.
"""
function reactant_pullback_pooled(∂A::AbstractMatrix{T},
                                   Rnl_3::AbstractArray{T,3},
                                   Ylm_3::AbstractArray{T,3},
                                   spec_R::AbstractVector{<:Integer},
                                   spec_Y::AbstractVector{<:Integer}) where {T}
    maxneigs, nnodes, nRnl = size(Rnl_3)
    _, _, nYlm = size(Ylm_3)
    nA = length(spec_R)

    # Initialize gradients
    ∂Rnl = zeros(T, maxneigs, nnodes, nRnl)
    ∂Ylm = zeros(T, maxneigs, nnodes, nYlm)

    # Expand ∂A for broadcasting: (nnodes, nA) -> (maxneigs, nnodes, nA)
    ∂A_expanded = reshape(∂A, 1, nnodes, nA)

    # Gather embeddings
    Rnl_gathered = Rnl_3[:, :, spec_R]  # (maxneigs, nnodes, nA)
    Ylm_gathered = Ylm_3[:, :, spec_Y]  # (maxneigs, nnodes, nA)

    # Compute gradients (∂A * other_factor for each)
    ∂prod_R = ∂A_expanded .* Ylm_gathered  # ∂L/∂(Rnl at spec_R)
    ∂prod_Y = ∂A_expanded .* Rnl_gathered  # ∂L/∂(Ylm at spec_Y)

    # Scatter-add back to full gradient arrays
    # This is the tricky part - need to accumulate gradients for repeated indices
    for iA in 1:nA
        iR = spec_R[iA]
        iY = spec_Y[iA]
        ∂Rnl[:, :, iR] .+= ∂prod_R[:, :, iA]
        ∂Ylm[:, :, iY] .+= ∂prod_Y[:, :, iA]
    end

    return ∂Rnl, ∂Ylm
end

# ========================================================================
# Conversion utilities
# ========================================================================

"""
    spec_to_arrays(spec::Vector{NTuple{2, Int}})

Convert Vector{Tuple} spec to separate integer arrays for Reactant compatibility.
"""
function spec_to_arrays(spec::Vector{NTuple{2, Int}})
    spec_R = [s[1] for s in spec]
    spec_Y = [s[2] for s in spec]
    return spec_R, spec_Y
end

"""
    aaspecs_to_matrices(aaspecs)

Convert tuple-based aaspecs to matrix format for Reactant compatibility.
Each aaspec[i] is a Vector of NTuples representing products of A indices.
"""
function aaspecs_to_matrices(aaspecs)
    return [_aaspec_to_matrix(aa) for aa in aaspecs]
end

function _aaspec_to_matrix(aaspec::Vector{<:Tuple})
    if isempty(aaspec)
        return zeros(Int, 0, 0)
    end
    order = length(first(aaspec))
    mat = zeros(Int, length(aaspec), order)
    for (i, aa) in enumerate(aaspec)
        for (j, idx) in enumerate(aa)
            mat[i, j] = idx
        end
    end
    return mat
end

# ========================================================================
# Reactant-compatible reshape_embedding
#
# Converts 2D (nedges, nfeatures) to 3D (maxneigs, nnodes, nfeatures)
# using pre-computed index mapping instead of KernelAbstractions
# ========================================================================

import EquivariantTensors: ETGraph, reshape_embedding, rev_reshape_embedding, nnodes, nedges, maxneigs

"""
    reactant_reshape_embedding(P, first_arr, maxn, nn)

Reactant-compatible version of reshape_embedding.

Arguments:
- P: (nedges, nfeatures) - input 2D array
- first_arr: (nnodes+1,) - cumulative edge counts (first[i] = start index for node i)
- maxn: maximum neighbors per node
- nn: number of nodes

Returns:
- P3: (maxneigs, nnodes, nfeatures) - reshaped 3D array
"""
function reactant_reshape_embedding(P::AbstractMatrix{T},
                                     first_arr::AbstractVector{<:Integer},
                                     maxn::Integer,
                                     nn::Integer) where {T}
    nedg, nfeatures = size(P)

    # Build index mapping: edge_idx_map[t, inode] = edge index or 0 (for padding)
    # This is done at compile time (constant) so can use scalar indexing
    edge_idx_map = zeros(Int, maxn, nn)
    for inode in 1:nn
        i1 = first_arr[inode]
        i2 = first_arr[inode + 1] - 1
        num_neigs = i2 - i1 + 1
        for t in 1:num_neigs
            edge_idx_map[t, inode] = i1 + t - 1
        end
    end

    # Flatten the mapping for gather
    flat_idx = vec(edge_idx_map)  # (maxn * nn,)

    # Add a dummy row at the end of P for zero padding
    P_padded = vcat(P, zeros(T, 1, nfeatures))  # (nedges+1, nfeatures)

    # Replace 0 indices with nedges+1 (the zero-padding row)
    flat_idx_safe = replace(flat_idx, 0 => nedg + 1)

    # Gather: get P_padded[flat_idx_safe, :] -> (maxn*nn, nfeatures)
    P_gathered = P_padded[flat_idx_safe, :]

    # Reshape to 3D: (maxn, nn, nfeatures)
    P3 = reshape(P_gathered, maxn, nn, nfeatures)

    return P3
end

"""
    reactant_rev_reshape_embedding(P3, first_arr, nedg)

Reactant-compatible reverse of reshape_embedding.

Arguments:
- P3: (maxneigs, nnodes, nfeatures) - 3D array
- first_arr: (nnodes+1,) - cumulative edge counts
- nedg: number of edges

Returns:
- P: (nedges, nfeatures) - 2D array
"""
function reactant_rev_reshape_embedding(P3::AbstractArray{T,3},
                                         first_arr::AbstractVector{<:Integer},
                                         nedg::Integer) where {T}
    maxn, nn, nfeatures = size(P3)

    # Build reverse index mapping: for each edge, which (t, inode) location?
    # edge_to_tidx[iedge] = t, edge_to_node[iedge] = inode
    edge_to_tidx = zeros(Int, nedg)
    edge_to_node = zeros(Int, nedg)
    for inode in 1:nn
        i1 = first_arr[inode]
        i2 = first_arr[inode + 1] - 1
        for t in 1:(i2 - i1 + 1)
            iedge = i1 + t - 1
            edge_to_tidx[iedge] = t
            edge_to_node[iedge] = inode
        end
    end

    # Gather from P3 using the reverse mapping
    # P[iedge, :] = P3[edge_to_tidx[iedge], edge_to_node[iedge], :]
    P = similar(P3, nedg, nfeatures)
    for ifeat in 1:nfeatures
        for iedge in 1:nedg
            P[iedge, ifeat] = P3[edge_to_tidx[iedge], edge_to_node[iedge], ifeat]
        end
    end

    return P
end

# ========================================================================
# Override Reactant-specific implementations
# ========================================================================

function EquivariantTensors._reactant_reshape_embedding(P::AbstractMatrix, X::ETGraph)
    return reactant_reshape_embedding(P, X.first, maxneigs(X), nnodes(X))
end

function EquivariantTensors._reactant_rev_reshape_embedding(P3::AbstractArray{T,3}, X::ETGraph) where {T}
    return reactant_rev_reshape_embedding(P3, X.first, nedges(X))
end

# ========================================================================
# Reactant-compatible SelectLinL (readout layer)
# ========================================================================

import EquivariantTensors: SelectLinL, _apply_selectlinl, _reactant_apply_selectlinl

"""
    reactant_apply_selectlinl(P, species_idx, W)

Reactant-compatible SelectLinL application for single-batch evaluation.

Arguments:
- P: (nnodes, n_basis) - input basis values
- species_idx: (nnodes,) - species index for each node (1-indexed)
- W: (out_dim, n_basis, n_species) - weight tensor

Returns:
- B: (nnodes, out_dim) - output
"""
function reactant_apply_selectlinl(P::AbstractMatrix,
                                    species_idx::AbstractVector,
                                    W::AbstractArray{<:Any, 3})
    nnodes, n_basis = size(P)
    out_dim, _, n_species = size(W)

    # Infer output type from multiplication of element types
    TB = typeof(zero(eltype(P)) * zero(eltype(W)))

    # Use broadcasting/element-wise ops to avoid BLAS (which doesn't work with ConcreteRArrays)
    # B[i, j] = sum_k P[i, k] * W[j, k, species_idx[i]]
    # Strategy: For each species, compute contribution and mask

    # Initialize output
    B = zeros(TB, nnodes, out_dim)

    # For each species, compute masked contribution
    for s in 1:n_species
        # Mask for nodes of this species: (nnodes,)
        mask = species_idx .== s

        # Weight for this species: (out_dim, n_basis)
        Ws = W[:, :, s]

        # Compute P @ Ws' using broadcasting to avoid BLAS
        # P: (nnodes, n_basis), Ws: (out_dim, n_basis)
        # Want: (nnodes, out_dim) where B[i,j] = sum_k P[i,k] * Ws[j,k]
        # Expand P to (nnodes, 1, n_basis) and Ws to (1, out_dim, n_basis)
        P_exp = reshape(P, nnodes, 1, n_basis)
        Ws_exp = reshape(Ws, 1, out_dim, n_basis)

        # Element-wise multiply and sum over last dimension
        P_times_W = dropdims(sum(P_exp .* Ws_exp, dims=3), dims=3)  # (nnodes, out_dim)

        # Add masked contribution using broadcasting
        mask_exp = reshape(mask, nnodes, 1)  # (nnodes, 1)
        B = B .+ mask_exp .* P_times_W
    end

    return B
end

# Override the main dispatch function
# For Reactant, X is expected to be an integer array of species indices directly
# rather than NamedTuples requiring selector application
function EquivariantTensors._reactant_apply_selectlinl(l::SelectLinL, P, X::AbstractVector{<:Integer}, W)
    return reactant_apply_selectlinl(P, X, W)
end

# Specific dispatch for TracedRArray of integers
# TracedRArray{Int64, 1} has element type TracedRNumber{Int64}, not Int64, so
# it doesn't match AbstractVector{<:Integer}
function EquivariantTensors._reactant_apply_selectlinl(l::SelectLinL, P, X::TracedRArray{T, 1}, W) where {T<:Integer}
    return reactant_apply_selectlinl(P, X, W)
end

# Same for ConcreteRArray
function EquivariantTensors._reactant_apply_selectlinl(l::SelectLinL, P, X::AbstractConcreteArray{T, 1}, W) where {T<:Integer}
    return reactant_apply_selectlinl(P, X, W)
end

# For NamedTuple arrays, try to apply selector first (may not trace if selector uses scalar indexing)
function EquivariantTensors._reactant_apply_selectlinl(l::SelectLinL, P, X::AbstractVector, W)
    # Apply selector to get species indices
    species_idx = [l.selector(x) for x in X]
    return reactant_apply_selectlinl(P, species_idx, W)
end

# ========================================================================
# Dynamic Topology Path (Alternative)
#
# These functions accept topology arrays (first_arr, etc.) as traced inputs
# rather than extracting from ETGraph at trace time. This enables variable
# topology where ii, jj change at runtime.
#
# Key difference: index mappings are computed using vectorized ops that
# Reactant can trace, rather than scalar loops.
# ========================================================================

"""
    compute_edge_index_map(first_arr, maxn, nn)

Compute edge index mapping for 2D→3D reshape using vectorized operations.
This is Reactant-traceable (no scalar loops).

Arguments:
- first_arr: (nn+1,) cumulative edge counts (can be TracedRArray)
- maxn: maximum neighbors per node (Int constant)
- nn: number of nodes (Int constant)

Returns:
- edge_idx_map: (maxn, nn) where edge_idx_map[t, i] = edge index or 0
- valid_mask: (maxn, nn) boolean mask for valid entries

Note: Uses AbstractVector (not AbstractVector{<:Integer}) to match TracedRArray.
"""
function compute_edge_index_map(first_arr::AbstractVector,
                                 maxn::Integer,
                                 nn::Integer)
    # Compute number of neighbors per node: (nn,)
    num_neigs = first_arr[2:nn+1] .- first_arr[1:nn]

    # Create t indices: (maxn,) - use Int64 for concrete type
    t_indices = Int64.(1:maxn)

    # Create base offsets from first_arr: (nn,)
    base_offsets = first_arr[1:nn]

    # Compute raw edge indices using broadcasting: (maxn, nn)
    # edge_idx[t, i] = first_arr[i] + t - 1
    edge_idx_raw = base_offsets' .+ t_indices .- Int64(1)

    # Create validity mask: t <= num_neigs[i]
    valid_mask = t_indices .<= num_neigs'

    # Zero out invalid entries using element-wise multiply
    edge_idx_map = edge_idx_raw .* valid_mask

    return edge_idx_map, valid_mask
end

"""
    compute_first_array(ii, n_atoms)

Compute cumulative edge count array from edge source indices.
Uses vectorized histogram-like computation.

NOTE: This function works in native Julia but does NOT trace well in Reactant
due to cumsum+vcat patterns. For Reactant export, compute first_arr on the
host (Python) and pass it as an input to the VMFB:

    # Python
    counts = np.bincount(ii - 1, minlength=n_atoms)  # ii is 1-based
    first_arr = np.cumsum(np.concatenate([[0], counts])) + 1

Arguments:
- ii: (n_edges,) source node index for each edge (1-based)
- n_atoms: number of nodes

Returns:
- first_arr: (n_atoms+1,) where first_arr[i] = start index for node i's edges
"""
function compute_first_array(ii::AbstractVector, n_atoms::Integer)
    n_edges = length(ii)

    # Count edges per node using one-hot encoding + sum
    # Use Int64 for indices
    node_indices = Int64.(1:n_atoms)'  # (1, n_atoms)
    ii_expanded = reshape(ii, n_edges, 1)  # (n_edges, 1)

    # indicator[e, i] = (ii[e] == i) as integer
    indicator = Int64.(ii_expanded .== node_indices)  # (n_edges, n_atoms)

    # Sum to get edge count per node: (n_atoms,)
    counts = dropdims(sum(indicator, dims=1), dims=1)

    # Cumulative sum to get first array
    # Build without mutation: first_arr = cumsum([1; counts])
    # This works in native Julia but may not trace in Reactant
    cs = cumsum(counts)
    first_arr = vcat([Int64(1)], cs .+ Int64(1))

    return first_arr
end

"""
    reshape_embedding_dynamic(P, first_arr, maxn, nn, n_edges)

Reactant-traceable 2D→3D reshape with dynamic topology.

Unlike `reactant_reshape_embedding`, this version computes the index mapping
using vectorized operations that Reactant can trace, enabling variable topology
where first_arr can be a runtime input.

Arguments:
- P: (n_edges, n_features) input embeddings (can be TracedRArray)
- first_arr: (nn+1,) cumulative edge counts (can be TracedRArray)
- maxn: maximum neighbors (Int constant)
- nn: number of nodes (Int constant for shape)
- n_edges: number of edges (Int constant for shape)

Returns:
- P3: (maxn, nn, n_features) reshaped embeddings
"""
function reshape_embedding_dynamic(P::AbstractMatrix,
                                    first_arr::AbstractVector,
                                    maxn::Integer,
                                    nn::Integer,
                                    n_edges::Integer)
    n_features = size(P, 2)

    # Compute edge index mapping (vectorized, traceable)
    edge_idx_map, valid_mask = compute_edge_index_map(first_arr, maxn, nn)

    # Flatten for gather operation
    flat_idx = vec(edge_idx_map)  # (maxn * nn,)
    flat_mask = vec(valid_mask)   # (maxn * nn,)

    # Add padding row to P for zero-gather
    T = eltype(P)
    P_padded = vcat(P, zeros(T, 1, n_features))  # (n_edges+1, n_features)

    # Replace 0 indices with n_edges+1 (padding row)
    # Use: idx_safe = valid * idx + (1 - valid) * (n_edges + 1)
    flat_mask_int = Int64.(flat_mask)
    idx_safe = flat_mask_int .* flat_idx .+ (Int64(1) .- flat_mask_int) .* Int64(n_edges + 1)

    # Gather: (maxn*nn, n_features)
    P_gathered = P_padded[idx_safe, :]

    # Reshape to 3D
    P3 = reshape(P_gathered, maxn, nn, n_features)

    return P3
end

"""
    rev_reshape_embedding_dynamic(P3, first_arr, n_edges)

Reverse 3D→2D reshape for dynamic topology.

NOTE: This function does NOT trace in Reactant due to data-dependent indexing
(`first_arr[node_for_edge]`). For Reactant export, use Enzyme autodiff which
automatically differentiates through `reshape_embedding_dynamic`.

Arguments:
- P3: (maxn, nn, n_features) input
- first_arr: (nn+1,) cumulative edge counts (native Julia Vector)
- n_edges: number of edges

Returns:
- P: (n_edges, n_features) output
"""
function rev_reshape_embedding_dynamic(P3::AbstractArray{T,3},
                                        first_arr::AbstractVector,
                                        n_edges::Integer) where {T}
    maxn, nn, n_features = size(P3)

    # Build reverse mapping: for each edge, which (t, node) location?
    edge_indices = Int64.(collect(1:n_edges))  # (n_edges,) concrete array

    # For each edge, find which node it belongs to
    node_starts = first_arr[1:nn]'  # (1, nn)
    edge_expanded = reshape(edge_indices, n_edges, 1)  # (n_edges, 1)

    # For each edge, count how many node starts are <= edge index
    # This gives the node index (1-based)
    belongs_to = sum(Int64.(edge_expanded .>= node_starts), dims=2)  # (n_edges, 1)
    node_for_edge = dropdims(belongs_to, dims=2)  # (n_edges,)

    # Compute t for each edge: t = edge - first_arr[node] + 1
    # This requires indexing first_arr with node_for_edge (data-dependent)
    t_for_edge = edge_indices .- first_arr[node_for_edge] .+ Int64(1)

    # Linear index into P3 (column-major): idx = t + (node-1)*maxn
    linear_base = t_for_edge .+ (node_for_edge .- Int64(1)) .* Int64(maxn)

    # Gather from flattened P3
    P3_flat = reshape(P3, maxn * nn, n_features)
    P = P3_flat[linear_base, :]

    return P
end

# ========================================================================
# Alternative dispatch for dynamic topology
# These use AbstractVector for first_arr to match traced arrays
# ========================================================================

# Export the dynamic functions for use in ACEpotentials
export compute_edge_index_map, compute_first_array
export reshape_embedding_dynamic, rev_reshape_embedding_dynamic

# ========================================================================
# Vectorized Embedding Evaluation for Reactant
#
# These functions provide Reactant-traceable evaluation paths that avoid
# scalar indexing. They accept arrays (rij, zi, zj) instead of Vector{XState}.
# ========================================================================

import EquivariantTensors: EmbedDP, EdgeEmbed, TransSelSplines, NTtransformST, SelectLinL
import EquivariantTensors: cat2idx, catcat2idx, symidx

# -------------------------------------------------------------------------
# Vectorized Transform Evaluation
# -------------------------------------------------------------------------

"""
    reactant_compute_distances(rij::AbstractMatrix)

Compute distances and unit vectors from displacement matrix.

Arguments:
- rij: (n_edges, 3) displacement vectors

Returns:
- r: (n_edges,) distances
- rhat: (n_edges, 3) unit vectors
"""
function reactant_compute_distances(rij::AbstractMatrix{T}) where {T}
    # rij: (n_edges, 3)
    r_sq = sum(rij .^ 2, dims=2)  # (n_edges, 1)
    r = sqrt.(r_sq .+ T(1e-12))   # safe sqrt to avoid zero
    rhat = rij ./ r               # (n_edges, 3) unit vectors
    return dropdims(r, dims=2), rhat
end

"""
    reactant_apply_agnesi_transform(r, zi, zj, params, zlist)

Vectorized Agnesi transform for Reactant.

Arguments:
- r: (n_edges,) distances
- zi: (n_edges,) center species indices (1-based into zlist)
- zj: (n_edges,) neighbor species indices (1-based into zlist)
- params: Vector of NamedTuple params for each species pair
- zlist: list of species (for symmetric indexing)

Returns:
- y: (n_edges,) transformed distances in [-1, 1]
"""
function reactant_apply_agnesi_transform(r::AbstractVector{T},
                                          zi::AbstractVector{<:Integer},
                                          zj::AbstractVector{<:Integer},
                                          params, zlist) where {T}
    n_edges = length(r)
    NZ = length(zlist)

    # Get concrete type for array allocation
    T_concrete = isbitstype(T) ? T : eltype(T)

    # Initialize output with concrete type
    y = zeros(T_concrete, n_edges)

    # Loop over species pairs (small constant number)
    # For symmetric storage: params[symidx(i, j, NZ)] for i <= j
    for i in 1:NZ, j in i:NZ
        # Create mask for this species pair (symmetric)
        mask_ij = (zi .== i) .& (zj .== j)
        mask_ji = (zi .== j) .& (zj .== i)
        mask = mask_ij .| mask_ji

        # Get parameters for this pair
        idx = symidx(i, j, NZ)
        p = params[idx]

        # Vectorized Agnesi evaluation
        rin, req = p.rin, p.req
        pin, pcut = p.pin, p.pcut
        a, b0, b1 = p.a, p.b0, p.b1

        s = (r .- rin) ./ (req - rin)
        x = 1 ./ (1 .+ a .* s.^pin ./ (1 .+ s.^(pin - pcut)))
        y_val = clamp.(b1 .* x .+ b0, T(-1), T(1))

        # Add masked contribution
        y = y .+ mask .* y_val
    end

    return y
end

# -------------------------------------------------------------------------
# Vectorized Spline Evaluation (no KernelAbstractions)
# -------------------------------------------------------------------------

"""
    reactant_eval_cubic_vectorized(t, fl, fr, gl, gr)

Vectorized cubic spline evaluation.

Arguments (all same size or broadcastable):
- t: position in [0,1]
- fl, fr: function values at left/right
- gl, gr: gradients at left/right (scaled by h)

Returns:
- s: spline values
"""
function reactant_eval_cubic_vectorized(t, fl, fr, gl, gr)
    # Cubic Hermite interpolation
    # s(t) = (2t³ - 3t² + 1)fl + (t³ - 2t² + t)gl + (-2t³ + 3t²)fr + (t³ - t²)gr
    a0 = fl
    a1 = gl
    a2 = -3 .* fl .+ 3 .* fr .- 2 .* gl .- gr
    a3 = 2 .* fl .- 2 .* fr .+ gl .+ gr
    return ((a3 .* t .+ a2) .* t .+ a1) .* t .+ a0
end

"""
    reactant_apply_transsplines(y, sel_idx, st_params, n_basis)

Vectorized spline evaluation for Reactant using global polynomial approximation.

Instead of one-hot cell selection (which creates large intermediate matrices),
this version fits a polynomial approximation to the entire spline function.
This trades exact spline evaluation for much faster compilation.

Arguments:
- y: (n_edges,) transformed distances
- sel_idx: (n_edges,) species-pair selector indices (1-based)
- st_params: NamedTuple with F, G, x0, x1 arrays from TransSelSplines state
- n_basis: output dimension (number of basis functions per edge)

Returns:
- S: (n_edges, n_basis) spline values
"""
function reactant_apply_transsplines(y::AbstractVector{T},
                                      sel_idx::AbstractVector{<:Integer},
                                      st_params, n_basis::Integer) where {T}
    n_edges = length(y)
    n_cat = size(st_params.F, 2)  # number of species categories
    NX = size(st_params.F, 1)     # number of spline grid points

    # Polynomial degree for approximation
    # Higher degree = more accuracy but more computation
    # Degree 2*NX-3 would match cubic spline exactly at knots, but that's too high
    # Use a moderate degree that provides good approximation
    poly_deg = min(2 * NX - 1, 15)  # Cap at degree 15

    # Pre-compute polynomial coefficients by fitting to spline values
    # We'll sample the spline at many points and fit a polynomial
    n_sample = max(poly_deg + 1, 50)

    # Coefficient matrix: (poly_deg+1, n_basis, n_cat)
    poly_coeffs = zeros(Float64, poly_deg + 1, n_basis, n_cat)
    x0_arr = zeros(Float64, n_cat)
    x1_arr = zeros(Float64, n_cat)

    for icat in 1:n_cat
        x0 = st_params.x0[icat]
        x1 = st_params.x1[icat]
        x0_arr[icat] = x0
        x1_arr[icat] = x1
        h = (x1 - x0) / (NX - 1)

        F_cat = st_params.F[:, icat]  # (NX,) of SVector{n_basis}
        G_cat = st_params.G[:, icat]  # (NX,) of SVector{n_basis}

        # Sample points for polynomial fitting (in normalized [-1, 1] space)
        y_sample = range(x0 + 1e-8, x1 - 1e-8, length=n_sample)

        # Evaluate original spline at sample points
        spline_vals = zeros(Float64, n_sample, n_basis)
        for (is, ys) in enumerate(y_sample)
            # Find cell and local t
            t_global = (ys - x0) / h
            icell = min(Int(floor(t_global)), NX - 2) + 1  # 1-based, clamped
            t_local = t_global - (icell - 1)

            for ib in 1:n_basis
                fl = F_cat[icell][ib]
                fr = F_cat[icell + 1][ib]
                gl = G_cat[icell][ib] * h
                gr = G_cat[icell + 1][ib] * h

                # Cubic Hermite: s(t) = a0 + a1*t + a2*t² + a3*t³
                a0 = fl
                a1 = gl
                a2 = -3*fl + 3*fr - 2*gl - gr
                a3 = 2*fl - 2*fr + gl + gr
                spline_vals[is, ib] = a0 + a1*t_local + a2*t_local^2 + a3*t_local^3
            end
        end

        # Fit polynomial using normalized coordinates
        # Normalize y to [-1, 1] for numerical stability
        y_norm = 2.0 .* (y_sample .- x0) ./ (x1 - x0) .- 1.0

        # Build Vandermonde matrix
        V = zeros(Float64, n_sample, poly_deg + 1)
        for is in 1:n_sample
            for k in 0:poly_deg
                V[is, k+1] = y_norm[is]^k
            end
        end

        # Solve least squares for each basis function
        for ib in 1:n_basis
            # V * c = spline_vals[:, ib]
            # c = V \ spline_vals
            poly_coeffs[:, ib, icat] = V \ spline_vals[:, ib]
        end
    end

    # Initialize output using traced-compatible zeros
    y_col = reshape(y, n_edges, 1)
    S = y_col .* zeros(Float64, 1, n_basis) .* 0.0

    # Process each species category
    for icat in 1:n_cat
        mask = sel_idx .== icat
        mask_float = mask .* 1.0
        mask_expanded = reshape(mask_float, n_edges, 1)

        x0 = x0_arr[icat]
        x1 = x1_arr[icat]

        # Normalize y to [-1, 1] for this category
        y_clamped = clamp.(y, x0, x1)
        y_norm = 2.0 .* (y_clamped .- x0) ./ (x1 - x0) .- 1.0

        # Evaluate polynomial using Horner's method
        # P(y) = c_0 + c_1*y + c_2*y² + ... + c_D*y^D
        #      = c_0 + y*(c_1 + y*(c_2 + ... + y*c_D))

        coeffs_cat = poly_coeffs[:, :, icat]  # (poly_deg+1, n_basis)

        # Start with highest degree coefficient
        S_cat = reshape(coeffs_cat[end, :], 1, n_basis) .* ones(Float64, n_edges, 1)

        # Horner's method: work down from highest to lowest degree
        y_norm_col = reshape(y_norm, n_edges, 1)
        for k in (poly_deg-1):-1:0
            c_k = reshape(coeffs_cat[k+1, :], 1, n_basis)
            S_cat = S_cat .* y_norm_col .+ c_k
        end

        S = S .+ mask_expanded .* S_cat
    end

    return S
end

# -------------------------------------------------------------------------
# Vectorized EmbedDP
# -------------------------------------------------------------------------

"""
    reactant_compute_selector_idx(zi, zj, NZ)

Compute species-pair selector index for (zi, zj) pairs.
Matches the indexing: idx = (i-1)*NZ + j for non-symmetric, or symidx for symmetric.

Arguments:
- zi: (n_edges,) center species indices (1-based)
- zj: (n_edges,) neighbor species indices (1-based)
- NZ: number of species

Returns:
- sel_idx: (n_edges,) selector indices
"""
function reactant_compute_selector_idx(zi::AbstractVector{<:Integer},
                                        zj::AbstractVector{<:Integer},
                                        NZ::Integer)
    # Non-symmetric indexing: idx = (i-1)*NZ + j
    return (zi .- 1) .* NZ .+ zj
end

"""
    reactant_compute_selector_idx_sym(zi, zj, NZ)

Compute symmetric species-pair selector index.
Uses upper-triangular storage: symidx(min(i,j), max(i,j), NZ)

Arguments:
- zi, zj: (n_edges,) species indices (1-based)
- NZ: number of species

Returns:
- sel_idx: (n_edges,) selector indices for symmetric storage
"""
function reactant_compute_selector_idx_sym(zi::AbstractVector{<:Integer},
                                            zj::AbstractVector{<:Integer},
                                            NZ::Integer)
    n_edges = length(zi)
    sel_idx = zeros(Int64, n_edges)

    # Loop over all possible symmetric pairs
    for i in 1:NZ, j in i:NZ
        idx = symidx(i, j, NZ)
        mask = ((zi .== i) .& (zj .== j)) .| ((zi .== j) .& (zj .== i))
        sel_idx .+= mask .* idx
    end

    return sel_idx
end

"""
    reactant_apply_embeddp_radial(rij, zi, zj, trans_params, zlist,
                                   spline_st, W, n_basis)

Vectorized radial EmbedDP for Reactant (with SelectLinL post).

This handles the common case: EmbedDP(agnesi_trans, basis, SelectLinL)
or TransSelSplines (which bakes SelectLinL into splines).

Arguments:
- rij: (n_edges, 3) displacement vectors
- zi: (n_edges,) center species indices (1-based into zlist)
- zj: (n_edges,) neighbor species indices (1-based into zlist)
- trans_params: Agnesi transform parameters
- zlist: list of species
- spline_st: spline state (F, G, x0, x1) or nothing if using polynomials
- W: weight tensor for SelectLinL (out_dim, in_dim, n_cat) or nothing for splines
- n_basis: output dimension

Returns:
- Rnl: (n_edges, n_basis) radial embeddings
"""
function reactant_apply_embeddp_radial(rij::AbstractMatrix{T},
                                        zi::AbstractVector{<:Integer},
                                        zj::AbstractVector{<:Integer},
                                        trans_params, zlist,
                                        spline_st, W, n_basis::Integer) where {T}
    n_edges = size(rij, 1)
    NZ = length(zlist)

    # Step 1: Compute distances
    r, rhat = reactant_compute_distances(rij)

    # Step 2: Apply transform (Agnesi)
    y = reactant_apply_agnesi_transform(r, zi, zj, trans_params, zlist)

    # Step 3: Apply splines or polynomials + SelectLinL
    if spline_st !== nothing
        # TransSelSplines case: splines already include SelectLinL weights
        sel_idx = reactant_compute_selector_idx_sym(zi, zj, NZ)
        Rnl = reactant_apply_transsplines(y, sel_idx, spline_st, n_basis)
    else
        error("Non-spline radial basis not yet implemented for Reactant")
    end

    return Rnl
end

# -------------------------------------------------------------------------
# Vectorized Spherical Harmonics (Angular Embedding)
# -------------------------------------------------------------------------

"""
    reactant_apply_spherical_harmonics(rhat, maxl)

Vectorized real spherical harmonics evaluation.

Arguments:
- rhat: (n_edges, 3) unit vectors
- maxl: maximum angular momentum

Returns:
- Ylm: (n_edges, n_ylm) spherical harmonics values
"""
function reactant_apply_spherical_harmonics(rhat::AbstractMatrix{T}, maxl::Integer) where {T}
    n_edges = size(rhat, 1)

    # Extract components
    x = rhat[:, 1]
    y = rhat[:, 2]
    z = rhat[:, 3]

    # Compute l=0 (1 function)
    # Y_0^0 = 1/(2√π)
    c0 = T(0.5) / sqrt(T(π))
    Y00 = fill(c0, n_edges)

    if maxl == 0
        return reshape(Y00, n_edges, 1)
    end

    # Compute l=1 (3 functions)
    # Y_1^{-1} = √(3/4π) y
    # Y_1^0 = √(3/4π) z
    # Y_1^1 = √(3/4π) x
    c1 = sqrt(T(3) / (4 * T(π)))
    Y1m1 = c1 .* y
    Y10 = c1 .* z
    Y11 = c1 .* x

    if maxl == 1
        return hcat(Y00, Y1m1, Y10, Y11)
    end

    # Compute l=2 (5 functions) - real spherical harmonics
    c2_0 = sqrt(T(5) / (16 * T(π)))      # for Y_2^0
    c2_1 = sqrt(T(15) / (4 * T(π)))       # for Y_2^{±1}
    c2_2 = sqrt(T(15) / (16 * T(π)))      # for Y_2^{±2}

    Y2m2 = c2_2 .* x .* y                              # xy
    Y2m1 = c2_1 .* y .* z                              # yz
    Y20 = c2_0 .* (2 .* z.^2 .- x.^2 .- y.^2)          # 2z²-x²-y²
    Y21 = c2_1 .* z .* x                               # zx
    Y22 = c2_2 .* (x.^2 .- y.^2)                       # x²-y²

    if maxl == 2
        return hcat(Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22)
    end

    # For higher l, would need to implement more terms
    error("maxl > 2 not yet implemented in vectorized spherical harmonics")
end

"""
    reactant_apply_embeddp_angular(rij, maxl)

Vectorized angular EmbedDP for Reactant.
This handles yembed which just extracts rhat and applies spherical harmonics.

Arguments:
- rij: (n_edges, 3) displacement vectors
- maxl: maximum angular momentum

Returns:
- Ylm: (n_edges, n_ylm) spherical harmonics values
"""
function reactant_apply_embeddp_angular(rij::AbstractMatrix{T}, maxl::Integer) where {T}
    # Compute unit vectors
    _, rhat = reactant_compute_distances(rij)

    # Apply spherical harmonics
    return reactant_apply_spherical_harmonics(rhat, maxl)
end

# -------------------------------------------------------------------------
# Vectorized EdgeEmbed (2D→3D reshape)
# -------------------------------------------------------------------------

"""
    reactant_edge_embed_radial(rij, zi, zj, first_arr, maxn, nn, n_edges,
                                trans_params, zlist, spline_st, W, n_basis)

Full vectorized radial EdgeEmbed for Reactant.
Computes 2D embeddings then reshapes to 3D.

Arguments:
- rij: (n_edges, 3) displacement vectors
- zi, zj: (n_edges,) species indices
- first_arr: (nn+1,) cumulative edge counts
- maxn: max neighbors
- nn: number of nodes
- n_edges: number of edges
- trans_params, zlist, spline_st, W, n_basis: embedding parameters

Returns:
- Rnl_3d: (maxn, nn, n_basis) 3D radial embeddings
"""
function reactant_edge_embed_radial(rij, zi, zj, first_arr, maxn, nn, n_edges,
                                     trans_params, zlist, spline_st, W, n_basis)
    # Step 1: Compute 2D embeddings
    Rnl_2d = reactant_apply_embeddp_radial(rij, zi, zj, trans_params, zlist,
                                            spline_st, W, n_basis)

    # Step 2: Reshape to 3D
    Rnl_3d = reshape_embedding_dynamic(Rnl_2d, first_arr, maxn, nn, n_edges)

    return Rnl_3d
end

"""
    reactant_edge_embed_angular(rij, first_arr, maxn, nn, n_edges, maxl)

Full vectorized angular EdgeEmbed for Reactant.

Arguments:
- rij: (n_edges, 3) displacement vectors
- first_arr: (nn+1,) cumulative edge counts
- maxn, nn, n_edges: topology dimensions
- maxl: maximum angular momentum

Returns:
- Ylm_3d: (maxn, nn, n_ylm) 3D angular embeddings
"""
function reactant_edge_embed_angular(rij, first_arr, maxn, nn, n_edges, maxl)
    # Step 1: Compute 2D embeddings
    Ylm_2d = reactant_apply_embeddp_angular(rij, maxl)

    # Step 2: Reshape to 3D
    Ylm_3d = reshape_embedding_dynamic(Ylm_2d, first_arr, maxn, nn, n_edges)

    return Ylm_3d
end

# =========================================================================
# Full ETACE Evaluation from rij
#
# This function provides a complete Reactant-traceable path for evaluating
# ETACE models from displacement vectors.
# =========================================================================

"""
    extract_radial_params(rembed, st_rembed)

Extract parameters needed for vectorized radial embedding evaluation.

Returns:
- trans_params: Agnesi transform parameters (zlist, params)
- spline_st: Spline state (F, G, x0, x1)
- n_basis: Number of output basis functions
"""
function extract_radial_params(rembed::EdgeEmbed, st_rembed)
    layer = rembed.layer

    if layer isa EmbedDP && layer.basis isa TransSelSplines
        # EmbedDP with TransSelSplines (spline basis with built-in SelectLinL)
        trans_splines = layer.basis

        # Trans params from the inner transform's state
        trans_st = st_rembed.trans
        trans_params = trans_st.params
        zlist = trans_st.zlist

        # Spline state
        spline_st = st_rembed.basis.params

        # Get n_basis from spline F shape
        n_basis = length(spline_st.F[1, 1])  # F[grid_idx, cat_idx] is SVector{n_basis}

        return (; trans_params, zlist, spline_st, n_basis)
    elseif layer isa TransSelSplines
        # Direct TransSelSplines layer
        trans_st = st_rembed.trans
        trans_params = trans_st.params
        zlist = trans_st.zlist
        spline_st = st_rembed.params
        n_basis = length(spline_st.F[1, 1])

        return (; trans_params, zlist, spline_st, n_basis)
    else
        error("Unsupported radial embedding layer type: $(typeof(layer))")
    end
end

"""
    extract_angular_params(yembed, st_yembed)

Extract parameters needed for vectorized angular embedding evaluation.

Returns:
- maxl: Maximum angular momentum
"""
function extract_angular_params(yembed::EdgeEmbed, st_yembed)
    layer = yembed.layer

    # The angular layer typically uses P4ML spherical harmonics
    # We need to extract maxl from the basis specification
    # This is a simplification - assumes standard Ylm embedding

    # Try to get maxl from the basis shape
    if hasfield(typeof(layer), :basis)
        basis = layer.basis
        if hasmethod(Base.length, (typeof(basis),))
            n_ylm = length(basis)
            # n_ylm = (maxl+1)^2, so maxl = sqrt(n_ylm) - 1
            maxl = Int(sqrt(n_ylm)) - 1
            return (; maxl)
        end
    end

    # Fallback: use maxl=2 as default (9 spherical harmonics)
    return (; maxl=2)
end

"""
    reactant_apply_etace(rembed, yembed, basis, readout,
                          rij, zi, zj, first_arr, maxn, nn, n_edges,
                          node_species, ps, st)

Full ETACE evaluation from displacement vectors using vectorized code paths.

This is the Reactant-compatible replacement for the standard ETACE evaluation
that uses scalar indexing over edge_data.

Arguments:
- rembed: Radial EdgeEmbed layer
- yembed: Angular EdgeEmbed layer
- basis: ACE basis layer
- readout: SelectLinL readout layer
- rij: (n_edges, 3) displacement vectors (can be TracedRArray)
- zi: (n_edges,) center species indices (1-based)
- zj: (n_edges,) neighbor species indices (1-based)
- first_arr: (nn+1,) cumulative edge counts
- maxn: Maximum neighbors per node
- nn: Number of nodes
- n_edges: Number of edges
- node_species: (nn,) species index for each node
- ps: Model parameters
- st: Model state

Returns:
- energy: Total potential energy (scalar)
"""
function reactant_apply_etace(rembed, yembed, basis, readout,
                               rij, zi, zj, first_arr, maxn, nn, n_edges,
                               node_species, ps, st)
    # Extract embedding parameters
    radial_params = extract_radial_params(rembed, st.rembed)
    angular_params = extract_angular_params(yembed, st.yembed)

    # Compute radial embeddings (2D then reshape to 3D)
    Rnl_3d = reactant_edge_embed_radial(
        rij, zi, zj, first_arr, maxn, nn, n_edges,
        radial_params.trans_params, radial_params.zlist,
        radial_params.spline_st, nothing, radial_params.n_basis
    )

    # Compute angular embeddings (2D then reshape to 3D)
    Ylm_3d = reactant_edge_embed_angular(
        rij, first_arr, maxn, nn, n_edges, angular_params.maxl
    )

    # ACE basis evaluation (already works with 3D inputs)
    (BB,), _ = basis((Rnl_3d, Ylm_3d), ps.basis, st.basis)

    # Readout (use vectorized SelectLinL)
    W = ps.readout.W
    phi = reactant_apply_selectlinl(BB, node_species, W)

    # Sum to get total energy
    return sum(phi)
end

"""
    reactant_apply_etace_with_embeddings(basis, readout,
                                          Rnl_3d, Ylm_3d,
                                          node_species, ps_basis, st_basis, W)

ETACE evaluation from pre-computed 3D embeddings.
This is useful when embeddings are computed separately (e.g., from Julia host).

Arguments:
- basis: ACE basis layer
- readout: SelectLinL readout layer
- Rnl_3d: (maxn, nn, n_rnl) 3D radial embeddings
- Ylm_3d: (maxn, nn, n_ylm) 3D angular embeddings
- node_species: (nn,) species index for each node
- ps_basis: Basis parameters
- st_basis: Basis state
- W: Readout weights (out_dim, n_basis, n_species)

Returns:
- energy: Total potential energy (scalar)
"""
function reactant_apply_etace_with_embeddings(basis, readout,
                                               Rnl_3d, Ylm_3d,
                                               node_species, ps_basis, st_basis, W)
    # ACE basis evaluation
    (BB,), _ = basis((Rnl_3d, Ylm_3d), ps_basis, st_basis)

    # Readout
    phi = reactant_apply_selectlinl(BB, node_species, W)

    # Sum to get total energy
    return sum(phi)
end

# Export the vectorized embedding functions
export reactant_compute_distances, reactant_apply_agnesi_transform
export reactant_apply_transsplines, reactant_eval_cubic_vectorized
export reactant_compute_selector_idx, reactant_compute_selector_idx_sym
export reactant_apply_embeddp_radial, reactant_apply_embeddp_angular
export reactant_apply_spherical_harmonics
export reactant_edge_embed_radial, reactant_edge_embed_angular
export extract_radial_params, extract_angular_params
export reactant_apply_etace, reactant_apply_etace_with_embeddings

end # module
