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

end # module
