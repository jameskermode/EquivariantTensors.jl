#
# KernelAbstractions evaluation of a sparse ACE model
#

using LinearAlgebra: transpose
import ChainRulesCore: rrule

# NOTES:
#  - Rnl and Ylm must be 3-dimensional arrays; cf. SparseProdPool for
#    the format.
#  - this kernel is inconsistent with the agreed-format for the 𝔹 basis
#    which is supposed to be returned as a tuple. But for initial testing,
#    this is ok.

function ka_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, ps, st)
   # Check for Reactant TracedRArrays - route to pure Julia path
   if _is_reactant_traced(Rnl_3) || _is_reactant_traced(Ylm_3)
      return _reactant_evaluate(tensor, Rnl_3, Ylm_3, ps, st)
   end

   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3,
                          st.aspec, st.aaspecs, st.A2Bmaps)
   return 𝔹, st
end

# ========================================================================
# Reactant-compatible pure Julia evaluation path
# Uses integer spec arrays instead of Vector{Tuple}
# ========================================================================

function _reactant_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, ps, st)
   # Use integer spec arrays from state
   spec_R = st.spec_R
   spec_Y = st.spec_Y
   aaspecs_mats = st.aaspecs_mats
   # Use dense A2Bmaps for Reactant (sparse versions use KA kernels which don't trace)
   A2Bmaps_dense = st.A2Bmaps_dense

   # Step 1: Pooled sparse product using integer arrays
   # A = (nnodes, nA)
   A = _reactant_pooled_sparse_product(Rnl_3, Ylm_3, spec_R, spec_Y)

   # Step 2: Sparse symmetric product for each order
   # AA = (nnodes, nAA)
   AA = _reactant_sparse_symm_prod(A, aaspecs_mats)

   # Step 3: Apply A2Bmaps (coupling coefficients) using dense matrices
   # 𝔹 = tuple of (nnodes, nB) matrices
   # Use standard matrix multiplication: AA @ A2Bmap' = (nnodes, nAA) @ (nAA, nB) = (nnodes, nB)
   # Each A2Bmap is (nB, nAA), so we compute AA * A2Bmap'
   𝔹 = Tuple(AA * A2Bmap' for A2Bmap in A2Bmaps_dense)

   return 𝔹, st
end

"""
Reactant-compatible pooled sparse product (NB=2 case).
Uses integer arrays spec_R, spec_Y instead of Vector{Tuple}.

Note: spec_R/spec_Y are typed as AbstractVector (not AbstractVector{<:Integer})
to support TracedRArray{Int64} which has element type TracedRNumber{Int64},
not Int64 itself.
"""
function _reactant_pooled_sparse_product(Rnl_3::AbstractArray{T,3},
                                          Ylm_3::AbstractArray{T,3},
                                          spec_R::AbstractVector,
                                          spec_Y::AbstractVector) where {T}
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
Reactant-compatible sparse symmetric product.
Handles multi-order products using matrix-format specs.
"""
function _reactant_sparse_symm_prod(A::AbstractMatrix{T},
                                     aaspecs_mats::Vector) where {T}
   nnodes = size(A, 1)

   # Get concrete type for array allocation
   # For traced types like TracedRNumber{Float32}, get Float32
   # isbitstype check: concrete types like Float32 are bits types
   T_concrete = isbitstype(T) ? T : eltype(T)

   # Compute AA for each order
   AA_parts = Vector{AbstractMatrix}()

   for aaspec_mat in aaspecs_mats
      if size(aaspec_mat, 1) == 0
         continue
      end
      AA_part = _reactant_symm_prod_single(A, aaspec_mat, T_concrete)
      push!(AA_parts, AA_part)
   end

   # Concatenate all orders
   if isempty(AA_parts)
      return zeros(T_concrete, nnodes, 0)
   else
      return hcat(AA_parts...)
   end
end

"""
Single-order symmetric product using matrix-format spec.
aaspec_mat is (nAA, order) matrix of A indices.

Note: aaspec_mat typed as AbstractMatrix (not AbstractMatrix{<:Integer})
to support TracedRArray which has TracedRNumber element type.
"""
function _reactant_symm_prod_single(A::AbstractMatrix{T},
                                     aaspec_mat::AbstractMatrix,
                                     T_concrete::Type=T) where {T}
   nAA, order = size(aaspec_mat)
   nnodes = size(A, 1)

   if order == 0
      # Constant (order-0): just ones - use concrete type
      return ones(T_concrete, nnodes, nAA)
   elseif order == 1
      # Linear: just gather from A
      return A[:, aaspec_mat[:, 1]]
   else
      # Higher order: product of gathered terms
      # Start with first term (avoids needing ones())
      AA = A[:, aaspec_mat[:, 1]]
      for k in 2:order
         AA = AA .* A[:, aaspec_mat[:, k]]
      end
      return AA
   end
end                           

function _ka_evaluate(tensor::SparseACEbasis, Rnl_3, Ylm_3, 
                      aspec, aaspecs, A2Bmaps)
   # A = #nodes x #features
   A = ka_evaluate(tensor.abasis, (Rnl_3, Ylm_3), aspec)
   # AA = #nodes x #features 
   AA = ka_evaluate(tensor.aabasis, A, aaspecs)
   # BB = #nodes x #features (TODO: undo the double-transpose!!!)
   BB = permutedims.( mul.(A2Bmaps, Ref(transpose(AA))) )
   return BB, A, AA
end 


function _ka_pullback(∂𝔹, tensor::SparseACEbasis, Rnl_3, Ylm_3, A, AA, 
                      aspec, aaspecs, A2Bmaps)
   # 𝔹 is a tuple of bases, so ∂𝔹 is a tuple of tangents, which is 
   # managed as a ChainRulesCore.Tangent. (usually thunked) By 
   # extracting them as ∂𝔹[i] we get the tangent for the ith element 
   # of the forward pass. 

   # Each 𝔹[i] is of the following form:  
   #      𝔹 = (𝒞 * 𝔸')' = 𝔸 * 𝒞' 
   #      ∂𝔹 : 𝔹 = (∂𝔹 * 𝒞) : 𝔸
   #  =>  ∇_𝔸 (∂𝔹 : 𝔹) = ∂𝔹 * 𝒞

   ∂𝔸 = sum( mul(∂𝔹[i], A2Bmaps[i], (a, b) -> sum(a .* b)) for i = 1:length(A2Bmaps) )
   ∂A = ka_pullback(∂𝔸, tensor.aabasis, A, aaspecs)
   ∂Rnl, ∂Ylm = ka_pullback(∂A, tensor.abasis, (Rnl_3, Ylm_3), aspec)
   return ∂Rnl, ∂Ylm
end 




#
# this rrule is just a wrapper for _ka_pullback
#
function rrule(::typeof(_ka_evaluate), tensor::SparseACEbasis, 
               Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl_3, Ylm_3, aspec, aaspecs, A2Bmaps)

   function _pb(∂𝔹A𝔸)
      ∂𝔹 = ∂𝔹A𝔸[1]
      # ∂𝔹A𝔸[2] == ∂𝔹A𝔸[2] == ZeroTangent() because A and 𝔸 are just 
      # intermediates that we keep to accelerate the backprop, but are not 
      # actually returned! 
      if !(∂𝔹A𝔸[2] == ∂𝔹A𝔸[3] == ZeroTangent())
         error("rrule for _ka_evaluate requires that only ∂𝔹 ≠ 0")
      end

      ∂Rnl, ∂Ylm = _ka_pullback(∂𝔹, tensor, Rnl_3, Ylm_3, A, 𝔸, 
                                aspec, aaspecs, A2Bmaps)
      return (∂Rnl, ∂Ylm, )
   end

   return (𝔹, A, 𝔸), ∂𝔹A𝔸 -> (NoTangent(), NoTangent(), _pb(∂𝔹A𝔸)..., 
                              NoTangent(), NoTangent(), NoTangent())
end
