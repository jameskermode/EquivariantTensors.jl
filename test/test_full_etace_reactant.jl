# Test full ETACE model compilation with Reactant
#
# Run with: julia +1.11 --project=<ACEpotentials.export> test/test_full_etace_reactant.jl

using Test
using Reactant
Reactant.set_default_backend("cpu")
using ACEpotentials
using AtomsBase
using AtomsBase: ChemicalSpecies
using Lux, Random
using StaticArrays, Unitful
using LinearAlgebra: I
import EquivariantTensors as ET

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

@testset "Full ETACE Model Reactant Compilation" begin
    # Create minimal model
    rng = Random.MersenneTwister(42)
    elements = (:Si,)
    rcut = 5.5

    rin0cuts = M._default_rin0cuts(elements)
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

    model = M.ace_model(;
        elements = elements, order = 2, Ytype = :solid,
        level = M.TotalDegree(), max_level = 6, maxl = 2, pair_maxn = 6,
        rin0cuts = rin0cuts, pair_learnable = true
    )

    ps, st = Lux.setup(rng, model)
    calc = ETM.convert2et_full(model, ps, st; rng=rng)

    # Create reference system using periodic_system helper
    function create_ref_sys(elements, rcut)
        lattice_const = rcut * 0.6
        positions = [[0.0, 0.0, 0.0], [lattice_const/2, lattice_const/2, 0.0], [lattice_const/2, 0.0, lattice_const/2]]
        box = [[lattice_const, 0.0, 0.0], [0.0, lattice_const, 0.0], [0.0, 0.0, lattice_const]]
        return periodic_system(
            [:Si => pos * u"Å" for pos in positions],
            box * u"Å"
        )
    end

    sys = create_ref_sys(elements, rcut)
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    etace = calc.calcs[3]

    @info "Graph" nodes=ET.nnodes(G) edges=ET.nedges(G)

    # Reference energy
    result, _ = etace.model(G, etace.ps, etace.st)
    E_ref = sum(result)
    @info "Reference energy" E=E_ref

    @testset "Compile full model" begin
        # Create a closure that captures model and params
        model_ref = etace.model
        ps_ref = etace.ps
        st_ref = etace.st

        function energy_fn(graph)
            result, _ = model_ref(graph, ps_ref, st_ref)
            return sum(result)
        end

        # Try compilation
        compiled = Reactant.@compile energy_fn(G)
        E_compiled = compiled(G)

        @info "Compiled energy" E=E_compiled
        @test isapprox(E_ref, E_compiled; atol=1e-6)
    end

    @testset "Non-trivial energy" begin
        # Set some non-zero weights to get non-trivial energy
        # Modify readout weights
        ps_modified = deepcopy(etace.ps)
        ps_modified.readout.W .= 0.1  # Set to non-zero value

        model_ref = etace.model
        st_ref = etace.st

        # Reference energy with modified params
        result_ref, _ = model_ref(G, ps_modified, st_ref)
        E_ref_mod = sum(result_ref)
        @info "Modified reference energy" E=E_ref_mod
        @test abs(E_ref_mod) > 0.0  # Should be non-zero now

        # Compile with modified params
        function energy_fn_mod(graph)
            result, _ = model_ref(graph, ps_modified, st_ref)
            return sum(result)
        end

        compiled_mod = Reactant.@compile energy_fn_mod(G)
        E_compiled_mod = compiled_mod(G)

        @info "Modified compiled energy" E=E_compiled_mod
        @test isapprox(E_ref_mod, E_compiled_mod; rtol=1e-4)
    end
end
