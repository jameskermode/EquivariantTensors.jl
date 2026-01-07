#=
ACE Model Export to StableHLO
=============================

This script exports the Reactant-compatible ACE model to StableHLO format
for compilation with IREE and integration with LAMMPS.

Key features:
1. Composes energy + gradients into a single exportable function
2. Uses Enzyme.autodiff for reverse-mode AD
3. Exports via Reactant.Serialization.export_to_enzymejax()
4. Generates .mlir + .npz + Python wrapper

Usage:
    julia --project ace_export.jl

Requirements:
    - NPZ.jl for export_to_enzymejax
    - Reactant compiled with serializable=true
=#

using Printf
using LinearAlgebra
import EquivariantTensors as ET
import Polynomials4ML as P4ML
using SpheriCart
using Random
using Reactant
using Enzyme

# Include the Reactant-compatible ACE implementation
include("ace_reactant.jl")

println("="^70)
println("ACE MODEL EXPORT TO STABLEHLO")
println("="^70)

## ============================================================================
## Step 1: Build ACE Model
## ============================================================================

println("\n1. Building ACE model...")

# Model parameters (production-size)
Dtot, maxl, ORD = 6, 3, 3
N_cheb = Dtot + 1

mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

nfeatures = length(symbasis, 0)
@printf("Model: Dtot=%d, maxl=%d, ORD=%d, %d features\n", Dtot, maxl, ORD, nfeatures)

# Initialize state
rng = MersenneTwister(1234)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)
rst = prepare_reactant_state(st)

# Linear readout parameters
params = randn(Float32, nfeatures)

## ============================================================================
## Step 2: Create Test Data
## ============================================================================

println("\n2. Creating test data...")

maxneigs, nnodes = 15, 10
nRnl = N_cheb
nYlm = (maxl + 1)^2

Rnl_3 = randn(Float32, maxneigs, nnodes, nRnl)
Ylm_3 = randn(Float32, maxneigs, nnodes, nYlm)

@printf("   Rnl_3: %s, Ylm_3: %s\n", size(Rnl_3), size(Ylm_3))
@printf("   spec_R/Y: %d indices\n", length(rst.spec_R))
@printf("   specs_mats: %d matrices\n", length(rst.specs_mats))
@printf("   A2Bmap: %s\n", size(rst.A2Bmaps_dense[1]))
@printf("   params: %d\n", length(params))

## ============================================================================
## Step 3: Define Energy Function (for gradients)
## ============================================================================

println("\n3. Defining energy function...")

# Simple energy function that takes Rnl_3, Ylm_3 as primary inputs
# (the spec arrays are constants during inference)
function ace_energy_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
    BB, _, _ = ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    return sum(BB * params)
end

# Test energy computation
E_test = ace_energy_simple(Rnl_3, Ylm_3, rst.spec_R, rst.spec_Y,
                            rst.specs_mats, rst.A2Bmaps_dense[1], params)
@printf("   Test energy: %.6f\n", E_test)

## ============================================================================
## Step 4: Test Enzyme Gradients (Julia)
## ============================================================================

println("\n4. Testing Enzyme gradients (Julia)...")

# Allocate gradient buffers
dRnl_3 = zeros(Float32, size(Rnl_3))
dYlm_3 = zeros(Float32, size(Ylm_3))

# Compute gradients via Enzyme
# We differentiate w.r.t. Rnl_3 and Ylm_3 (the embedding outputs)
# These gradients will propagate back to position gradients in the full pipeline
try
    Enzyme.autodiff(
        Enzyme.Reverse,
        ace_energy_simple,
        Enzyme.Active,
        Enzyme.Duplicated(Rnl_3, dRnl_3),
        Enzyme.Duplicated(Ylm_3, dYlm_3),
        Enzyme.Const(rst.spec_R),
        Enzyme.Const(rst.spec_Y),
        Enzyme.Const(rst.specs_mats),
        Enzyme.Const(rst.A2Bmaps_dense[1]),
        Enzyme.Const(params)
    )

    @printf("   [OK] Enzyme gradients computed\n")
    @printf("   dRnl_3 norm: %.6f\n", norm(dRnl_3))
    @printf("   dYlm_3 norm: %.6f\n", norm(dYlm_3))

catch e
    println("   [FAIL] Enzyme gradient computation failed")
    showerror(stdout, e, catch_backtrace())
    println()
end

## ============================================================================
## Step 5: Define Combined Energy+Forces Function
## ============================================================================

println("\n5. Defining combined energy+forces function...")

# This function computes both energy and gradients in a single call
# Returns tuple: (energy, dRnl_3, dYlm_3)
function ace_energy_and_forces(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
    # Allocate gradient buffers
    dRnl = zero(Rnl_3)
    dYlm = zero(Ylm_3)

    # Forward pass: compute energy via autodiff with reverse mode
    # This computes both the energy and the gradients
    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        ace_energy_simple,
        Enzyme.Active,
        Enzyme.Duplicated(Rnl_3, dRnl),
        Enzyme.Duplicated(Ylm_3, dYlm),
        Enzyme.Const(spec_R),
        Enzyme.Const(spec_Y),
        Enzyme.Const(specs_mats),
        Enzyme.Const(A2Bmap),
        Enzyme.Const(params)
    )

    return (energy, dRnl, dYlm)
end

# Test combined function
try
    E_combined, dRnl_combined, dYlm_combined = ace_energy_and_forces(
        Rnl_3, Ylm_3, rst.spec_R, rst.spec_Y,
        rst.specs_mats, rst.A2Bmaps_dense[1], params
    )

    @printf("   [OK] Combined function works\n")
    @printf("   Energy: %.6f\n", E_combined)
    @printf("   dRnl norm: %.6f (expected: %.6f)\n", norm(dRnl_combined), norm(dRnl_3))
    @printf("   dYlm norm: %.6f (expected: %.6f)\n", norm(dYlm_combined), norm(dYlm_3))

catch e
    println("   [FAIL] Combined function failed")
    showerror(stdout, e, catch_backtrace())
    println()
end

## ============================================================================
## Step 6: Reactant Compilation Test
## ============================================================================

println("\n6. Testing Reactant compilation...")

try
    Reactant.set_default_backend("cpu")

    # Convert to RArrays
    Rnl_ra = Reactant.to_rarray(Rnl_3)
    Ylm_ra = Reactant.to_rarray(Ylm_3)
    spec_R_ra = Reactant.to_rarray(rst.spec_R)
    spec_Y_ra = Reactant.to_rarray(rst.spec_Y)
    specs_mats_ra = [Reactant.to_rarray(m) for m in rst.specs_mats]
    A2Bmap_ra = Reactant.to_rarray(rst.A2Bmaps_dense[1])
    params_ra = Reactant.to_rarray(params)

    println("   RArrays created...")

    # First test: just the energy function
    println("   Compiling energy function...")
    energy_compiled = Reactant.@compile ace_energy_simple(
        Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
        specs_mats_ra, A2Bmap_ra, params_ra
    )

    E_reactant = energy_compiled(Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
                                  specs_mats_ra, A2Bmap_ra, params_ra)
    @printf("   [OK] Energy compiled: %.6f (diff: %.2e)\n",
            Float64(E_reactant), abs(Float64(E_reactant) - E_test))

    # Second test: energy + forces
    println("   Compiling energy+forces function...")
    combined_compiled = Reactant.@compile ace_energy_and_forces(
        Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
        specs_mats_ra, A2Bmap_ra, params_ra
    )

    E_c, dRnl_c, dYlm_c = combined_compiled(Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
                                             specs_mats_ra, A2Bmap_ra, params_ra)
    @printf("   [OK] Energy+forces compiled: E=%.6f\n", Float64(E_c))
    @printf("        dRnl norm: %.6f, dYlm norm: %.6f\n",
            norm(Array(dRnl_c)), norm(Array(dYlm_c)))

catch e
    println("   [FAIL] Reactant compilation failed")
    showerror(stdout, e, catch_backtrace())
    println()
end

## ============================================================================
## Step 7: Export to StableHLO
## ============================================================================

println("\n7. Exporting to StableHLO...")

export_dir = joinpath(@__DIR__, "stablehlo_export")
mkpath(export_dir)

# Check if NPZ is available for export_to_enzymejax
npz_available = false
try
    using NPZ
    npz_available = true
    println("   NPZ.jl available for export")
catch e
    println("   NPZ.jl not available - will use @code_hlo instead")
end

if npz_available
    try
        Reactant.set_default_backend("cpu")

        # Prepare RArrays
        Rnl_ra = Reactant.to_rarray(Rnl_3)
        Ylm_ra = Reactant.to_rarray(Ylm_3)
        spec_R_ra = Reactant.to_rarray(rst.spec_R)
        spec_Y_ra = Reactant.to_rarray(rst.spec_Y)
        specs_mats_ra = [Reactant.to_rarray(m) for m in rst.specs_mats]
        A2Bmap_ra = Reactant.to_rarray(rst.A2Bmaps_dense[1])
        params_ra = Reactant.to_rarray(params)

        println("   Exporting via export_to_enzymejax...")

        # Export to MLIR + NPZ + Python wrapper
        Reactant.Serialization.export_to_enzymejax(
            ace_energy_simple,
            Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
            specs_mats_ra, A2Bmap_ra, params_ra;
            output_dir=export_dir,
            function_name="ace_energy"
        )

        println("   [OK] Export successful!")
        println("   Files created in: $export_dir")

        # List generated files
        for f in readdir(export_dir)
            path = joinpath(export_dir, f)
            @printf("      %s (%d bytes)\n", f, filesize(path))
        end

    catch e
        println("   [FAIL] export_to_enzymejax failed")
        showerror(stdout, e, catch_backtrace())
        println()
    end
else
    # Fallback: use @code_hlo to get StableHLO string
    println("   Using @code_hlo fallback...")

    try
        Reactant.set_default_backend("cpu")

        Rnl_ra = Reactant.to_rarray(Rnl_3)
        Ylm_ra = Reactant.to_rarray(Ylm_3)
        spec_R_ra = Reactant.to_rarray(rst.spec_R)
        spec_Y_ra = Reactant.to_rarray(rst.spec_Y)
        specs_mats_ra = [Reactant.to_rarray(m) for m in rst.specs_mats]
        A2Bmap_ra = Reactant.to_rarray(rst.A2Bmaps_dense[1])
        params_ra = Reactant.to_rarray(params)

        # Capture @code_hlo output
        hlo_path = joinpath(export_dir, "ace_energy.mlir")

        # Redirect stdout to capture @code_hlo output
        open(hlo_path, "w") do io
            redirect_stdout(io) do
                Reactant.@code_hlo ace_energy_simple(
                    Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
                    specs_mats_ra, A2Bmap_ra, params_ra
                )
            end
        end

        println("   [OK] StableHLO saved to: $hlo_path")
        @printf("      File size: %d bytes\n", filesize(hlo_path))

    catch e
        println("   [FAIL] @code_hlo fallback failed")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end

## ============================================================================
## Step 8: Save Model Constants
## ============================================================================

println("\n8. Saving model constants...")

constants_path = joinpath(export_dir, "ace_constants.jld2")

try
    using JLD2

    JLD2.@save constants_path begin
        spec_R = rst.spec_R
        spec_Y = rst.spec_Y
        specs_mats = rst.specs_mats
        A2Bmap = rst.A2Bmaps_dense[1]
        params = params
        N_cheb = N_cheb
        maxl = maxl
        nRnl = nRnl
        nYlm = nYlm
        nfeatures = nfeatures
    end

    println("   [OK] Constants saved to: $constants_path")

catch e
    # JLD2 not available, save as Julia file
    constants_jl_path = joinpath(export_dir, "ace_constants.jl")

    open(constants_jl_path, "w") do io
        println(io, "# ACE Model Constants")
        println(io, "# Generated by ace_export.jl")
        println(io)
        println(io, "const spec_R = $(rst.spec_R)")
        println(io, "const spec_Y = $(rst.spec_Y)")
        println(io, "const N_cheb = $N_cheb")
        println(io, "const maxl = $maxl")
        println(io, "const nRnl = $nRnl")
        println(io, "const nYlm = $nYlm")
        println(io, "const nfeatures = $nfeatures")
        println(io)
        println(io, "# specs_mats and A2Bmap are too large for text format")
        println(io, "# Use NPZ or JLD2 for binary export")
    end

    println("   [INFO] JLD2 not available, saved partial constants to: $constants_jl_path")
end

## ============================================================================
## Summary
## ============================================================================

println("\n" * "="^70)
println("EXPORT SUMMARY")
println("="^70)

println("""

Files created in: $export_dir

Model parameters:
  - Dtot=$Dtot, maxl=$maxl, ORD=$ORD
  - $nfeatures features, $nRnl radial, $nYlm angular

Next steps:

1. Install NPZ.jl for full export:
   ] add NPZ

2. Compile with IREE:
   ~/iree/bin/iree-compile \\
       --iree-input-type=stablehlo \\
       --iree-hal-target-backends=llvm-cpu \\
       $export_dir/ace_energy.mlir \\
       -o $export_dir/ace_energy_cpu.vmfb

3. For GPU:
   ~/iree/bin/iree-compile \\
       --iree-input-type=stablehlo \\
       --iree-hal-target-backends=cuda \\
       $export_dir/ace_energy.mlir \\
       -o $export_dir/ace_energy_cuda.vmfb

4. Test with iree-run-module:
   ~/iree/bin/iree-run-module \\
       --device=local-task \\
       --module=$export_dir/ace_energy_cpu.vmfb \\
       --function=main
""")

println("="^70)
