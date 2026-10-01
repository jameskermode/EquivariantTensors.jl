using SparseArrays
using LinearAlgebra: norm 

"""
   symmetrisation_matrix(L, mb_spec; prune, kwargs...) -> 𝔸2𝔹, 𝔸_spec

Generates the symmetrization operator for a sparse ACE basis. The basis is 
# specified via the input `mb_spec`, which is a list of basis function 
specifications of the form 
```julia 
mb_spec = [ [(n=0, l=0), (n=1, l=0)], [(n=1, l=1), (n=2, l=1)], ... ] 
```
i.e. a `Vector{Vector{NL}}`. where `NL = @NamedTuple{n::Int, l::Int}`.

The parameter `L` determines the order of the ouput, e.g. L=0 for an invariant 
scalar, L = 1 for a vector, and so forth. 

The output is given in terms of a sparse matrix `𝔸2𝔹` in CCS format and a 
specification of the `𝔸` basis as a `Vector{Vector{NLM}}` where 
`NLM = @NamedTuple{n::Int, l::Int, m::Int}`. 

The remaining `kwargs...` (`PI`, `basis`, `nullspace_solver`, ...) are passed 
to `O3.coupling_coeffs`.
"""
function symmetrisation_matrix(L::Integer, mb_spec; 
                               prune = false, kwargs...)

   # for now assume a specific form of the mb_spec, namely 
   #   Vector{Vector{NT_NLM}}   
   #   where NT_NLM = typeof( (n = 0, l = 0) )

   # convert an element of 𝔸spec to nn, ll, mm, which is the format 
   # used by the coupling_coeffs function 
   function _vecnt2nnll(bb)
      nn = [ b.n for b in bb ]
      ll = [ b.l for b in bb ]
      return nn, ll
   end

   # extract all unique (nn, ll) blocks, since the (ll, mm) will only be used 
   # in generating the coupled / symmetrized basis functions
   nnll = unique(_vecnt2nnll.(mb_spec))

   # Now for each (nn, ll) block we can generate all possible invariant basis 
   # functions. We assemble the symmetrization operator in triplet format, 
   # which can conveniently account for double-counting of entries.

   # NB : HACK TO DISTINGUISH L = 0 and L > 0
   #      this should potentially be revisited in the future 
   #      in fact this might be a type-stability issue
   TVAL = L == 0 ? Float64 : SVector{2*L+1, Float64}
   irow = Int[]; jcol = Int[]; val = TVAL[]

   # counter for total number of equivariant basis functions
   𝔸spec = Vector{@NamedTuple{n::Int64, l::Int64, m::Int64}}[]
   num𝔹 = 0 
   num𝔸 = 0
   for (nn, ll) in nnll
      # here the kwargs... should be PI and basis 
      cc, MM = O3.coupling_coeffs(L, ll, nn; kwargs...)
      num_b = size(cc, 1)   
      if num_b == 0; continue; end
      # lookup the corresponding (nn, ll, mm) in the 𝔸 specification 
      # idx_𝔸 = Int[ inv_𝔸spec[ (nn, ll, mm) ] for mm in MM ] 
      idx_𝔸 = Int[] 
      for (_i,mm) in enumerate(MM) 
         # _i = (inv_𝔸spec[ (nn, ll, mm) ])::Int 
         push!(idx_𝔸, _i+num𝔸)
         push!(𝔸spec, [(n = nn[i], l = ll[i], m = mm[i]) for i = 1:length(mm)])
      end
      num𝔸 += length(MM)
      # add the new basis functions to the triplet format
      for q = 1:num_b 
         num𝔹 += 1
         for j = 1:length(idx_𝔸)
            push!(irow, num𝔹); push!(jcol, idx_𝔸[j]); push!(val, cc[q, j])
         end
      end
   end

   @assert num𝔸 == length(𝔸spec)
   # assemble the symmetrization operator in compressed column format 
   symm = sparse(irow, jcol, val, num𝔹, num𝔸) 

   # prune rows with all-zero entries (if there are any then print a warning 
   # because this indicates a bug in `coupling_coeffs`)
   i_nz_rows = findall(!iszero, sum(norm, symm; dims = 2)[:])
   if length(i_nz_rows) != num𝔹
      @warn("symmetrization matrix has all-zero rows; this indicates a bug in `coupling_coeffs`")
      symm = symm[i_nz_rows, :]
   end

   if prune 
      i_nz_cols = sort( findall(!iszero, sum(norm, symm; dims = 1)[:]) ) 
      𝔸spec_pruned = 𝔸spec[i_nz_cols]
      symm_pruned = symm[:, i_nz_cols]
   else
      𝔸spec_pruned = 𝔸spec 
      symm_pruned = symm
   end 

   return symm_pruned, 𝔸spec_pruned
end
