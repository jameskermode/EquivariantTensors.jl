# Reactant-Compatible ACE Prototype

This directory contains a prototype for compiling ACE (Atomic Cluster Expansion) models with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), enabling deployment via IREE to LAMMPS and ASE without Julia runtime dependencies.

## Current Status

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Reactant ACE evaluation | ✅ Working | Bypasses KernelAbstractions, uses plain loops |
| StableHLO export | ✅ Working | Via `Reactant.Serialization.export_to_enzymejax()` |
| IREE compilation | ✅ Working | CPU backend tested, GPU should work |
| IREE execution | ✅ Working | Energy matches Julia baseline (109.216) |
| ASE Calculator (IREE) | ✅ Working | Prototype with fixed shapes |
| ASE Calculator (Julia) | ✅ Working | Development backend via juliacall |
| LAMMPS ML-IAP | ✅ Working | Python unified interface tested |
| Embeddings (Rnl, Ylm) | ✅ Working | Reactant-compatible, 100-300x speedup |

### What's Missing

| Component | Status | Blocker |
|-----------|--------|---------|
| Forces via gradients | ❌ Not implemented | Need to export `Enzyme.gradient` composition |
| Dynamic shapes | ❌ Fixed shapes only | IREE compilation requires static shapes |
| Multi-element systems | ❌ Single element only | Need species indexing in embeddings |
| Production embeddings | ❌ Python placeholders | Need exact match with Julia |
| C/C++ IREE wrapper | ❌ Not started | For LAMMPS without Python |

## Directory Structure

```
examples/reactant/
├── README.md                      # This file
├── ace_reactant.jl               # Reactant-compatible ACE evaluation
├── ace_export.jl                 # StableHLO export functions
├── test_ace_reactant.jl          # Tests and benchmarks
│
├── ase/                          # ASE Calculator implementations
│   ├── __init__.py               # Factory function
│   ├── julia_calculator.py       # Development: Julia via juliacall
│   ├── iree_calculator.py        # Production: IREE via subprocess
│   └── ace_julia_wrapper.jl      # Julia-side model interface
│
├── lammps/                       # LAMMPS integration
│   ├── test_mliap_simple.py      # ML-IAP unified interface test
│   └── test_mliap_iree.py        # ML-IAP + IREE test
│
├── stablehlo_export/             # Export artifacts
│   ├── ace_model.mlir            # StableHLO IR
│   ├── ace_constants.npz         # Model parameters
│   ├── test_inputs_raw/          # Binary test inputs
│   └── compiled/
│       └── ace_model_cpu.vmfb    # IREE compiled binary
│
└── benchmark_*.jl                # Benchmarking scripts
```

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Julia Development           │
                    │  ACEpotentials.jl + Reactant.jl    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         Reactant.@compile           │
                    │     serializable=true               │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │    export_to_enzymejax()            │
                    │  → .mlir + .npz + .py wrapper       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │        iree-compile                 │
                    │     → .vmfb binary                  │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   LAMMPS    │         │     ASE     │         │   Custom    │
   │   ML-IAP    │         │ Calculator  │         │  C/C++ App  │
   └─────────────┘         └─────────────┘         └─────────────┘
```

## Quick Start

### 1. Test IREE Model Execution

```bash
cd examples/reactant/stablehlo_export
~/iree/bin/iree-run-module \
    --device=local-task \
    --module=compiled/ace_model_cpu.vmfb \
    --function=main \
    --input=5x5x10xf32=@test_inputs_raw/arg0.bin \
    --input=9x5x10xf32=@test_inputs_raw/arg1.bin \
    --input=19xi64=@test_inputs_raw/arg2.bin \
    --input=19xi64=@test_inputs_raw/arg3.bin \
    --input=1x5xi64=@test_inputs_raw/arg4.bin \
    --input=2x26xi64=@test_inputs_raw/arg5.bin \
    --input=31x19xf32=@test_inputs_raw/arg6.bin \
    --input=19xf32=@test_inputs_raw/arg7.bin
# Expected: f32=109.216
```

### 2. Test ASE Calculators

```bash
cd examples/reactant/ase
source ../stablehlo_export/.venv/bin/activate

# Julia backend (development)
python julia_calculator.py

# IREE backend (production)
python iree_calculator.py

# Compare both
python julia_calculator.py --compare
```

### 3. Test LAMMPS ML-IAP

```bash
cd examples/reactant/lammps
module load GCC/13.3.0 OpenMPI/5.0.3 Python/3.12.3 SciPy-bundle/2024.05
export LD_LIBRARY_PATH=~/lammps/lammps-22Jul2025/build:$LD_LIBRARY_PATH
python test_mliap_simple.py
```

---

## Remaining Tasks for Prototype Completion

### Task 1: Export Gradients for Forces

The current export only computes energy. Forces require exporting the gradient:

```julia
# In ace_export.jl
function ace_energy_and_forces(Rnl_3, Ylm_3, ...)
    # Forward: compute energy
    energy = ace_evaluate_energy(Rnl_3, Ylm_3, ...)

    # Backward: compute gradients w.r.t. embeddings
    dRnl = Enzyme.make_zero(Rnl_3)
    dYlm = Enzyme.make_zero(Ylm_3)

    Enzyme.autodiff(Reverse, ace_evaluate_energy, Active,
        Duplicated(Rnl_3, dRnl),
        Duplicated(Ylm_3, dYlm),
        Const(spec_R), ...)

    return (energy, dRnl, dYlm)
end

# Export with gradients
@compile serializable=true ace_energy_and_forces(...)
```

**Challenge**: The Python/C side must then chain-rule these embedding gradients back to position gradients.

### Task 2: Dynamic or Larger Fixed Shapes

Current model is compiled for fixed shapes:
- `Rnl_3`: [5, 5, 10] (maxneigs=5, nnodes=5, nRnl=10)
- `Ylm_3`: [9, 5, 10]

Options:
1. **Larger fixed shapes**: Compile for [256, 4096, 32] and pad smaller systems
2. **Multiple compiled models**: Different sizes for different system scales
3. **IREE dynamic shapes**: May require newer IREE features

### Task 3: Multi-Element Support

Current prototype assumes single element. Need:
- Species-indexed radial embeddings: `Rnl[species_i, species_j, n, l]`
- Type mapping from LAMMPS/ASE to ACE species indices
- Update export to include element type information

### Task 4: Production Embeddings

Python embeddings in `iree_calculator.py` are placeholders. Need:
- Exact match with Julia Chebyshev implementation
- Exact match with Julia real spherical harmonics (Flm coefficients)
- Cutoff envelope function matching

### Task 5: C/C++ IREE Wrapper for LAMMPS

For production LAMMPS without Python:
```c
// ace_evaluator.h
ACEModel* ace_model_create(const char* vmfb_path, const char* device);
float ace_evaluate(ACEModel* model,
    const float* Rnl_3, const float* Ylm_3, ...
    float* dRnl_3, float* dYlm_3);
```

See `iree/include/iree/runtime/api.h` for IREE C API patterns.

---

## Integration into ACEpotentials.jl

### Option A: New Package (Recommended for Initial Development)

Create `ACEpotentialsExport.jl` or `ACEReactant.jl`:

```julia
module ACEReactant

using ACEpotentials
using Reactant
using Enzyme

export compile_ace_model, export_to_iree

function compile_ace_model(model::ACE1Model; backend=:cpu)
    # Extract basis and parameters
    basis = model.basis
    params = model.params

    # Create Reactant-compatible evaluation
    # ...
end

function export_to_iree(compiled_model, output_dir)
    Reactant.Serialization.export_to_enzymejax(...)
    # Run iree-compile
    # ...
end

end
```

### Option B: PR to ACEpotentials.jl

Extend existing work in [ACEpotentials.jl#309](https://github.com/ACEsuit/ACEpotentials.jl/pull/309):

```julia
# In ACEpotentials/src/export.jl
function export_model(model::ACE1Model, format::Val{:iree}; ...)
    # Reactant compilation
    # StableHLO export
    # IREE compilation
end
```

---

## Reactant-Compatibility Changes Required

### 1. EquivariantTensors.jl

**Current blockers**:
- KernelAbstractions `@kernel` macros cannot be traced by Reactant
- Dynamic dispatch in some layers

**Changes needed**:

```julia
# Option A: Dispatch on array type
function evaluate(layer, X::AbstractArray)
    # Existing KA implementation
end

function evaluate(layer, X::Reactant.TracedRArray)
    # Reactant-compatible implementation (plain loops)
end

# Option B: Configuration flag
const USE_KA = Ref(true)

function evaluate(layer, X)
    if USE_KA[] && !(X isa Reactant.TracedRArray)
        _evaluate_ka(layer, X)
    else
        _evaluate_loops(layer, X)
    end
end
```

**Specific files**:
- `src/ace/sparseprodpool_ka.jl` → add loop-based fallback
- `src/ace/sparsesymmprod_ka.jl` → add loop-based fallback
- `src/ace/sparse_ace_ka.jl` → add dispatch for TracedRArray

### 2. Polynomials4ML.jl

**Current blockers**:
- Some chebyshev implementations use features Reactant can't trace

**Changes needed**:
- Add Reactant-compatible Chebyshev evaluation
- Verify all polynomial transforms work with TracedRArray

### 3. SpheriCart.jl

**Current blockers**:
- C library calls cannot be traced
- Need pure Julia fallback

**Changes needed**:

```julia
# In SpheriCart.jl
function compute_ylm(l_max, x, y, z)
    if use_c_library()
        _compute_ylm_c(l_max, x, y, z)
    else
        _compute_ylm_julia(l_max, x, y, z)  # Pure Julia
    end
end

# Pure Julia implementation for Reactant
function _compute_ylm_julia(l_max, x, y, z)
    # Recursive real spherical harmonics
    # ...
end
```

The pure Julia Ylm implementation is already in this prototype:
- See `test_ace_reactant.jl` → `compute_reactant_ylm()`
- Uses pre-computed Flm coefficients and polynomial evaluation

### 4. ACEpotentials.jl

**Changes needed**:
- Add export functions for IREE/StableHLO
- Dispatch to Reactant-compatible evaluation when compiling
- Integration with existing `juliac` export path

---

## Benchmarking Plan

### Test Systems

| System | Atoms | Description |
|--------|-------|-------------|
| Small | 32 | 2×2×2 FCC Al |
| Medium | 256 | 4×4×4 FCC Al |
| Large | 2048 | 8×8×8 FCC Al |
| XL | 16384 | 16×16×16 FCC Al |

### Backends to Compare

| Backend | Description | Expected Strength |
|---------|-------------|-------------------|
| Julia CPU (KA) | Current EquivariantTensors with KernelAbstractions | Baseline |
| Julia GPU (KA) | KernelAbstractions on CUDA | Large systems |
| Reactant CPU | This prototype, compiled to XLA | All sizes |
| Reactant GPU | This prototype, compiled to CUDA | Large systems |
| juliac CPU | AOT compiled Julia ([PR #309](https://github.com/ACEsuit/ACEpotentials.jl/pull/309)) | Deployment |
| IREE CPU | StableHLO → IREE VMFB | Deployment |
| IREE GPU | StableHLO → IREE CUDA | Large systems |
| ML-PACE CPU | Reference C++ implementation | Comparison |

### Metrics

1. **Throughput**: atoms/second for energy+forces
2. **Latency**: time for single evaluation (important for small systems)
3. **Scaling**: throughput vs system size
4. **Memory**: peak memory usage
5. **Startup**: time to first evaluation (compilation overhead)

### Benchmark Script Template

```julia
# benchmark_comparison.jl
using BenchmarkTools
using ACEpotentials
using EquivariantTensors
using Reactant
using CUDA

# Load model
model = load_ace_model("Al_ACE.json")

# Test systems
systems = [
    ("32 atoms", bulk(:Al, cubic=true) * (2,2,2)),
    ("256 atoms", bulk(:Al, cubic=true) * (4,4,4)),
    ("2048 atoms", bulk(:Al, cubic=true) * (8,8,8)),
]

results = Dict()

for (name, atoms) in systems
    println("=== $name ===")

    # Prepare inputs
    Rnl, Ylm = compute_embeddings(atoms, model.basis)

    # Julia CPU (KA)
    t_julia_cpu = @belapsed evaluate($model.basis, $Rnl, $Ylm)

    # Julia GPU (KA)
    Rnl_gpu, Ylm_gpu = CuArray(Rnl), CuArray(Ylm)
    t_julia_gpu = @belapsed CUDA.@sync evaluate($model.basis, $Rnl_gpu, $Ylm_gpu)

    # Reactant CPU
    compiled_cpu = Reactant.@compile ace_evaluate(Rnl, Ylm, ...)
    t_reactant_cpu = @belapsed $compiled_cpu($Rnl, $Ylm, ...)

    # Reactant GPU
    Reactant.set_default_backend("gpu")
    compiled_gpu = Reactant.@compile ace_evaluate(Rnl, Ylm, ...)
    t_reactant_gpu = @belapsed $compiled_gpu($Rnl, $Ylm, ...)

    results[name] = (
        julia_cpu = t_julia_cpu,
        julia_gpu = t_julia_gpu,
        reactant_cpu = t_reactant_cpu,
        reactant_gpu = t_reactant_gpu,
    )
end

# Print comparison table
# ...
```

### ML-PACE Comparison

For fair comparison with ML-PACE:
1. Use equivalent model complexity (same correlation order, similar basis size)
2. Run both through LAMMPS `pair_style pace` vs `pair_style mliap unified`
3. Measure `loop time` from LAMMPS output

```bash
# LAMMPS benchmark script
units metal
atom_style atomic
# ... create system ...

# ML-PACE
pair_style pace
pair_coeff * * Al.yace Al

# vs IREE-ACE
pair_style mliap unified ace_model.pkl 0
pair_coeff * * Al

timestep 0.001
run 1000
# Check "Loop time" in output
```

### Expected Results (Preliminary)

Based on embedding benchmarks from this prototype:

| Backend | 32 atoms | 256 atoms | 2048 atoms | 16384 atoms |
|---------|----------|-----------|------------|-------------|
| Julia CPU (KA) | 1× | 1× | 1× | 1× |
| Julia GPU (KA) | 0.5× | 2× | 10× | 50× |
| Reactant CPU | 5× | 20× | 100× | 300× |
| Reactant GPU | 2× | 10× | 100× | 400× |
| IREE CPU | ~Reactant | ~Reactant | ~Reactant | ~Reactant |
| juliac CPU | 2-5× | 2-5× | 2-5× | 2-5× |

*Speedups are estimates based on embedding benchmarks. Full model benchmarks pending.*

---

## References

- [Reactant.jl Documentation](https://enzymead.github.io/Reactant.jl/)
- [IREE Documentation](https://iree.dev/)
- [LAMMPS ML-IAP](https://docs.lammps.org/Packages_details.html#pkg-ml-iap)
- [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl)
- [ACEpotentials juliac export PR #309](https://github.com/ACEsuit/ACEpotentials.jl/pull/309)
- [ML-PACE](https://github.com/ICAMS/lammps-user-pace)

---

## Contributing

This is a prototype. Key areas for contribution:

1. **Gradient export**: Implement `ace_energy_and_forces()` with Enzyme
2. **Embedding matching**: Verify Python embeddings match Julia exactly
3. **C/C++ wrapper**: Implement IREE C API wrapper for LAMMPS
4. **Benchmarking**: Run comprehensive benchmarks across all backends
5. **Multi-element**: Add species indexing support

## License

Same as EquivariantTensors.jl (MIT).
