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

    @testset "reshape_embedding dispatch" begin
        # Test that regular arrays use KA path (no error when Reactant not loaded)
        nnodes, nedges, nfeat = 4, 10, 5
        maxneigs = 4

        # Create a mock ETGraph
        ii = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]  # sorted by node
        jj = [2, 3, 4, 1, 3, 4, 1, 2, 1, 2]
        G = ET.ETGraph(ii, jj)

        # Test reshape_embedding with regular arrays
        P2 = randn(nedges, nfeat)
        P3 = ET.reshape_embedding(P2, G)
        @test size(P3) == (ET.maxneigs(G), nnodes, nfeat)

        # Test rev_reshape_embedding
        P2_back = ET.rev_reshape_embedding(P3, G)
        @test size(P2_back) == (nedges, nfeat)
        @test P2 ≈ P2_back
    end

    @testset "SelectLinL dispatch" begin
        # Test that regular arrays use KA path
        in_dim, out_dim, ncat = 5, 3, 2
        rng = MersenneTwister(43)

        selector = x -> x  # Identity for integer inputs
        linl = ET.SelectLinL(in_dim, out_dim, ncat, selector)
        ps_linl = ET.LuxCore.initialparameters(rng, linl)
        st_linl = ET.LuxCore.initialstates(rng, linl)

        # Test with integer species indices
        nbatch = 10
        P = randn(nbatch, in_dim)
        X = rand(1:ncat, nbatch)  # Species indices

        B, _ = linl((P, X), ps_linl, st_linl)
        @test size(B) == (nbatch, out_dim)
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

        @testset "reshape_embedding with Reactant arrays" begin
            Reactant.set_default_backend("cpu")

            nnodes, nedges, nfeat = 4, 10, 5

            # Create a mock ETGraph
            ii = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
            jj = [2, 3, 4, 1, 3, 4, 1, 2, 1, 2]
            G = ET.ETGraph(ii, jj)

            # Test with regular arrays first
            P2 = randn(Float32, nedges, nfeat)
            P3_ref = ET.reshape_embedding(P2, G)

            # Test with ConcreteRArray (should use Reactant path)
            P2_ra = Reactant.to_rarray(P2)
            P3_ra = ET.reshape_embedding(P2_ra, G)
            @test size(P3_ra) == size(P3_ref)
            @test maximum(abs.(Array(P3_ra) .- P3_ref)) < 1e-6

            # Test rev_reshape_embedding
            P2_back_ra = ET.rev_reshape_embedding(P3_ra, G)
            @test size(P2_back_ra) == size(P2)
            @test maximum(abs.(Array(P2_back_ra) .- P2)) < 1e-6
        end

        @testset "SelectLinL with Reactant arrays" begin
            Reactant.set_default_backend("cpu")

            in_dim, out_dim, ncat = 5, 3, 2
            rng = MersenneTwister(44)

            selector = x -> x  # Identity for integer inputs
            linl = ET.SelectLinL(in_dim, out_dim, ncat, selector)
            ps_linl = ET.LuxCore.initialparameters(rng, linl)
            st_linl = ET.LuxCore.initialstates(rng, linl)

            nbatch = 10
            P = randn(Float32, nbatch, in_dim)
            X = rand(1:ncat, nbatch)

            # Reference with regular arrays
            B_ref, _ = linl((P, X), ps_linl, st_linl)

            # Test with Reactant arrays
            P_ra = Reactant.to_rarray(P)
            X_ra = Reactant.to_rarray(Int64.(X))
            W_ra = Reactant.to_rarray(Float32.(ps_linl.W))

            # Direct call to Reactant path
            B_ra = ET._reactant_apply_selectlinl(linl, P_ra, X_ra, W_ra)
            @test size(B_ra) == size(B_ref)
            @test maximum(abs.(Array(B_ra) .- Float32.(B_ref))) < 1e-5
        end
    else
        @info "Reactant not available, skipping Reactant-specific tests"
    end

end
