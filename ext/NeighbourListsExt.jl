
module NeighbourListsExt

using NeighbourLists
using AtomsBase
using AtomsBase: AbstractSystem, species, position, cell_vectors, periodicity
using Unitful: ustrip
import EquivariantTensors as ET
using DecoratedParticles: PState

# ============================================================================
# interaction_graph - main entry point (works with NeighbourLists 0.5+)
# ============================================================================

"""
    interaction_graph(sys::AbstractSystem, rcut)

Convert an AtomsBase system to an ETGraph using a cutoff radius.

# Arguments
- `sys`: AtomsBase system
- `rcut`: Cutoff radius (with Unitful units)

# Returns
- `ETGraph`
"""
function ET.Atoms.interaction_graph(sys::AbstractSystem, rcut)
    nlist = NeighbourLists.PairList(sys, rcut)
    return ET.Atoms.nlist2graph(nlist, sys)
end

# ============================================================================
# nlist2graph - convert PairList to ETGraph (unchanged from original)
# ============================================================================

function ET.Atoms.nlist2graph(nlist::NeighbourLists.PairList, sys::AbstractSystem)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first)
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ]
   S_i = [ species(sys, i) for i in ii ]
   S_j = [ species(sys, j) for j in jj ]
   X_ij = [ PState(𝐫 = 𝐫, z0 = si, z1 = sj, 𝐒 = shift)
            for (𝐫, si, sj, shift) in zip(R_ij, S_i, S_j, nlist.S) ]

   # for node data we use _only_ the atomic species for now so that we
   # don't even give the option of using position information directly.
   # ... until we sort out how to best handle this in ET.
   X_i = [ PState(𝐫 = ustrip.(position(sys, i)),
                  z = species(sys, i))
           for i = 1:length(sys) ]

   cell_vecs_u = cell_vectors(sys)
   cell_vecs = ntuple( i -> ustrip.(cell_vecs_u[i]),
                       length(cell_vecs_u) )

   sys_data = ( pbc = periodicity(sys),
               cell = cell_vecs
              )

   G = ET.ETGraph(ii, jj;
                  edge_data = X_ij,
                  node_data = X_i,
                  graph_data = sys_data)
   @assert G.first == first

   return G
end

# ============================================================================
# forces_from_edge_grads - unchanged from original
# ============================================================================

function ET.Atoms.forces_from_edge_grads(sys::AbstractSystem, G::ET.ETGraph, ∇E_edges)

   TFRC = typeof(∇E_edges[1].𝐫)
   F = zeros(TFRC, length(sys))

   for (i, j, e) in zip(G.ii, G.jj, ∇E_edges)
      F[i] -= e.𝐫
      F[j] += e.𝐫
   end

   return F
end

end
