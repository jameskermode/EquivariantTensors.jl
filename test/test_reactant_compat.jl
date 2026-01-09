# Test Reactant compatibility features in EquivariantTensors
#
# This test file validates:
# 1. _is_reactant_traced stub function exists
# 2. initialstates includes spec_R, spec_Y, aaspecs_mats fields
# 3. Pure Julia Reactant evaluation path works
# 4. ReactantExt extension loads and detects Reactant arrays (when Reactant available)
# 5. Reactant compilation succeeds (when Reactant available)

using EquivariantTensors
import EquivariantTensors as ET
import Polynomials4ML as P4ML
using Random, LinearAlgebra
using Test

# Helper to check if Reactant is available
const REACTANT_AVAILABLE = try
    @eval using Reactant
    true
catch
    false
end

@testset "Reactant Compatibility" begin

    # Build a simple ACE basis for testing
    Dtot, maxl, ORD = 4, 2, 2
    N_cheb = Dtot + 1

    mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
        level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

    symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
        Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
        Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
        basis=real)

    rng = MersenneTwister(42)
    ps = ET.LuxCore.initialparameters(rng, symbasis)
    st = ET.LuxCore.initialstates(rng, symbasis)

    @testset "_is_reactant_traced stub" begin
        # Stub should return false for regular arrays when Reactant not loaded
        @test ET._is_reactant_traced(randn(3,3)) == false
        @test ET._is_reactant_traced(1.0) == false
        @test ET._is_reactant_traced((randn(3,3), randn(3,3))) == false
    end

    @testset "initialstates fields" begin
        # New fields should exist
        @test haskey(st, :spec_R)
        @test haskey(st, :spec_Y)
        @test haskey(st, :aaspecs_mats)

        # spec_R and spec_Y should be integer vectors
        @test st.spec_R isa Vector{Int}
        @test st.spec_Y isa Vector{Int}
        @test length(st.spec_R) == length(st.aspec)
        @test length(st.spec_Y) == length(st.aspec)

        # Values should match aspec tuples
        for (i, s) in enumerate(st.aspec)
            @test st.spec_R[i] == s[1]
            @test st.spec_Y[i] == s[2]
        end

        # aaspecs_mats should be Vector of Matrix{Int}
        @test st.aaspecs_mats isa Vector
        for mat in st.aaspecs_mats
            @test mat isa Matrix{Int}
        end
    end

    @testset "Reactant evaluation path (pure Julia)" begin
        maxneigs, nnodes = 10, 4
        nRnl = N_cheb
        nYlm = (maxl + 1)^2

        Rnl_3 = randn(maxneigs, nnodes, nRnl)
        Ylm_3 = randn(maxneigs, nnodes, nYlm)

        # Test pooled sparse product
        A = ET._reactant_pooled_sparse_product(Rnl_3, Ylm_3, st.spec_R, st.spec_Y)
        @test size(A) == (nnodes, length(st.spec_R))

        # Test sparse symmetric product
        AA = ET._reactant_sparse_symm_prod(A, st.aaspecs_mats)
        @test size(AA, 1) == nnodes
        @test size(AA, 2) > 0

        # Test full ka_evaluate path (should use KA for regular arrays)
        BB_ka, _ = ET.ka_evaluate(symbasis, Rnl_3, Ylm_3, ps, st)
        @test length(BB_ka) == 1
        @test size(BB_ka[1], 1) == nnodes
    end

    # Tests that require Reactant
    if REACTANT_AVAILABLE
        @testset "ReactantExt extension" begin
            Reactant.set_default_backend("cpu")

            # Extension should detect Reactant arrays
            ra = Reactant.to_rarray(randn(Float32, 3, 3))
            @test ET._is_reactant_traced(ra) == true
            @test ET._is_reactant_traced((ra, ra)) == true

            # Regular arrays should still return false
            @test ET._is_reactant_traced(randn(3,3)) == false
        end

        @testset "Reactant compilation" begin
            Reactant.set_default_backend("cpu")

            maxneigs, nnodes = 10, 4
            nRnl = N_cheb
            nYlm = (maxl + 1)^2

            Rnl_3 = randn(Float32, maxneigs, nnodes, nRnl)
            Ylm_3 = randn(Float32, maxneigs, nnodes, nYlm)

            # Simple energy function using Reactant path
            function test_energy(Rnl_3, Ylm_3, spec_R, spec_Y)
                A = ET._reactant_pooled_sparse_product(Rnl_3, Ylm_3, spec_R, spec_Y)
                return sum(A)
            end

            # Create RArrays
            Rnl_ra = Reactant.to_rarray(Rnl_3)
            Ylm_ra = Reactant.to_rarray(Ylm_3)
            spec_R_ra = Reactant.to_rarray(Int64.(st.spec_R))
            spec_Y_ra = Reactant.to_rarray(Int64.(st.spec_Y))

            # Compile and test
            compiled = Reactant.@compile test_energy(Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra)
            E_c = compiled(Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra)
            E_j = test_energy(Rnl_3, Ylm_3, st.spec_R, st.spec_Y)

            @test abs(E_j - Float64(E_c)) < 1e-4
        end
    else
        @info "Reactant not available, skipping Reactant-specific tests"
    end

end
