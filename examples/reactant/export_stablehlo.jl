#=
Export ACE to StableHLO for IREE Compilation
=============================================

This script demonstrates exporting the Reactant-compiled ACE model to StableHLO,
which can then be compiled with IREE for deployment without Julia dependency.

Reactant provides several macros for code inspection:
- @code_hlo  : Print StableHLO representation
- @code_mhlo : Print MHLO representation
- @code_xla  : Print XLA HLO after optimization

For export, Reactant provides:
- export_to_enzymejax: Exports to .mlir file + .npz + Python wrapper

IREE Compilation:
  ~/iree/bin/iree-compile \
      --iree-input-type=stablehlo \
      --iree-hal-target-backends=llvm-cpu \
      ace_energy.mlir \
      -o ace_energy.vmfb

Usage:
  julia --project export_stablehlo.jl
=#

using Printf
using LinearAlgebra
import EquivariantTensors as ET
import Polynomials4ML as P4ML
using SpheriCart
using Random
using Reactant
using Reactant: @code_hlo, @code_mhlo, @code_xla

# Include the ACE implementation
include("ace_reactant.jl")

# Helper macro to capture stdout
macro capture_out(ex)
    quote
        old_stdout = stdout
        rd, wr = redirect_stdout()
        try
            $(esc(ex))
        finally
            redirect_stdout(old_stdout)
        end
        close(wr)
        read(rd, String)
    end
end

println("="^70)
println("STABLEHLO EXPORT FOR ACE MODEL")
println("="^70)

## ============================================================================
## Build a test model
## ============================================================================

println("\n1. Building ACE model...")

Dtot, maxl, ORD = 4, 2, 2  # Small model for testing
N_cheb = Dtot + 1

mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

nfeatures = length(symbasis, 0)
@printf("Model: Dtot=%d, maxl=%d, ORD=%d, %d features\n", Dtot, maxl, ORD, nfeatures)

rng = MersenneTwister(1234)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)
rst = prepare_reactant_state(st)

## ============================================================================
## Create test data
## ============================================================================

println("\n2. Creating test data...")

maxneigs, nnodes = 10, 5
nRnl = N_cheb
nYlm = (maxl + 1)^2

Rnl_3 = randn(Float32, maxneigs, nnodes, nRnl)
Ylm_3 = randn(Float32, maxneigs, nnodes, nYlm)
params = randn(Float32, nfeatures)

## ============================================================================
## Energy function for export
## ============================================================================

function ace_energy(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
    BB, _, _ = ace_evaluate_reactant_simple(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    return sum(BB * params)
end

## ============================================================================
## Inspect StableHLO representation
## ============================================================================

println("\n3. Generating StableHLO representation...")

Reactant.set_default_backend("cpu")

Rnl_ra = Reactant.to_rarray(Rnl_3)
Ylm_ra = Reactant.to_rarray(Ylm_3)
spec_R_ra = Reactant.to_rarray(rst.spec_R)
spec_Y_ra = Reactant.to_rarray(rst.spec_Y)
specs_mats_ra = [Reactant.to_rarray(m) for m in rst.specs_mats]
A2Bmap_ra = Reactant.to_rarray(rst.A2Bmaps_dense[1])
params_ra = Reactant.to_rarray(params)

println("\nStableHLO code (first 100 lines):")
println("-"^70)

# Capture @code_hlo output
hlo_output = @capture_out begin
    @code_hlo ace_energy(Rnl_ra, Ylm_ra,
                          spec_R_ra, spec_Y_ra,
                          specs_mats_ra, A2Bmap_ra, params_ra)
end

# Print first 100 lines
lines = split(hlo_output, '\n')
for (i, line) in enumerate(lines)
    if i > 100
        println("... ($(length(lines) - 100) more lines)")
        break
    end
    println(line)
end

## ============================================================================
## Export using Reactant.Serialization
## ============================================================================

println("\n" * "-"^70)
println("4. Export options:")
println("-"^70)

println("""
Reactant provides two export mechanisms:

A. export_to_enzymejax (requires NPZ.jl):
   - Generates .mlir file with StableHLO
   - Generates .npz file with input arrays
   - Generates Python wrapper script

   Usage:
   ```julia
   using NPZ
   Reactant.Serialization.export_to_enzymejax(
       compiled_fn,
       output_dir="./export",
       function_name="ace_energy"
   )
   ```

B. export_as_tf_saved_model (requires PythonCall):
   - Generates TensorFlow SavedModel
   - Can be used with TFLite, TensorFlow.js, TF Serving

   Usage:
   ```julia
   using PythonCall
   Reactant.Serialization.export_as_tf_saved_model(
       compiled_fn,
       saved_model_path="./saved_model",
       input_locations=(:input, :input, :input, ...)
   )
   ```

C. Direct MLIR export (manual):
   - Use @code_hlo to get StableHLO string
   - Save to .mlir file
   - Compile with IREE
""")

## ============================================================================
## Manual StableHLO file export
## ============================================================================

println("\n5. Manual StableHLO export...")

export_dir = joinpath(@__DIR__, "stablehlo_export")
mkpath(export_dir)

# Save the StableHLO to a file
mlir_path = joinpath(export_dir, "ace_energy.mlir")
open(mlir_path, "w") do f
    write(f, hlo_output)
end

println("   Saved StableHLO to: $mlir_path")
println("   File size: $(filesize(mlir_path)) bytes")

## ============================================================================
## IREE compilation instructions
## ============================================================================

println("\n" * "-"^70)
println("6. IREE Compilation:")
println("-"^70)

println("""
To compile with IREE:

# For CPU:
~/iree/bin/iree-compile \\
    --iree-input-type=stablehlo \\
    --iree-hal-target-backends=llvm-cpu \\
    $mlir_path \\
    -o $(export_dir)/ace_energy_cpu.vmfb

# For CUDA GPU:
~/iree/bin/iree-compile \\
    --iree-input-type=stablehlo \\
    --iree-hal-target-backends=cuda \\
    $mlir_path \\
    -o $(export_dir)/ace_energy_cuda.vmfb

# For AMD GPU (ROCM):
~/iree/bin/iree-compile \\
    --iree-input-type=stablehlo \\
    --iree-hal-target-backends=rocm \\
    $mlir_path \\
    -o $(export_dir)/ace_energy_rocm.vmfb

The resulting .vmfb file is a portable binary that can be loaded
by the IREE runtime without any Julia dependency.

To run the compiled module:
~/iree/bin/iree-run-module \\
    --device=local-task \\
    --module=$(export_dir)/ace_energy_cpu.vmfb \\
    --function=main \\
    --input="10x5x5xf32=@rnl_data.npy" \\
    --input="10x5x9xf32=@ylm_data.npy" \\
    ...
""")

## ============================================================================
## LAMMPS integration notes
## ============================================================================

println("\n" * "-"^70)
println("7. LAMMPS MLIAP Integration:")
println("-"^70)

println("""
For LAMMPS MLIAP unified interface integration:

1. IREE provides a C runtime API for loading and executing .vmfb modules
2. Create a C/C++ wrapper that:
   - Loads the IREE module
   - Converts LAMMPS neighbor list to edge arrays
   - Calls the ACE energy/force functions
   - Returns results to LAMMPS

3. Build as shared library implementing MLIAP interface:
   - MLIAPModel::compute_descriptors()
   - MLIAPModel::compute_forces()

Reference: https://docs.lammps.org/pair_mliap.html

The key advantage: The compiled IREE module contains NO Julia code,
making it fully portable and suitable for HPC deployment.
""")

println("\n" * "="^70)
println("DONE")
println("="^70)
