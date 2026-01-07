# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EquivariantTensors.jl provides tools to construct equivariant tensor layers for neural network models with O(3) symmetry. It is the backend for ACEsuit packages like ACEpotentials.jl and ACEhamiltonians.jl. The package integrates with Lux for neural networks, ChainRules/Zygote for autodiff, and KernelAbstractions for GPU support.

## Commands

### Testing
```bash
# Run full test suite
julia --project -e "using Pkg; Pkg.test()"

# Run a single test file interactively
julia --project
include("test/test_utils/utils_testO3.jl")  # Load test utilities first
include("test/ace/test_sparseprodpool.jl")   # Then run specific test
```

### Documentation
```bash
# Build docs locally
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
julia --project=docs docs/make.jl
```

### Benchmarking
Benchmarks are in `benchmark/` using BenchmarkTools.

## Architecture

### Core Protocol (src/generics.jl)
All layers inherit from `AbstractETLayer` and implement these operations:
- `evaluate!(output, layer, input)` - In-place forward pass
- `pullback!(∂input, ∂output, layer, input)` - Reverse-mode AD
- `pushforward!(layer, input, ∂input)` - Forward-mode AD (ForwardDiff)
- `pullback2!` - Second-order derivatives

Allocating versions (`evaluate`, `pullback`, etc.) are auto-generated. Layers are callable: `layer(args...)` calls `evaluate(layer, args...)`.

### Module Structure
- **src/ace/** - Core ACE (Atomic Cluster Expansion) layers
  - `sparseprodpool.jl` - PooledSparseProduct: fused tensor product + pooling (key building block)
  - `sparsesymmprod.jl` - SparseSymmProd: sparse symmetric products
  - `sparse_ace_basis.jl` - SparseACEbasis: complete ACE basis
  - `*_ka.jl` files - KernelAbstractions GPU variants

- **src/O3/** - SO(3) symmetry coupling
  - `O3.jl` - Generalized Clebsch-Gordan coefficients, coupling_coeffs
  - `yyvector.jl` - YY vector operations for real/complex basis
  - `O3_utils.jl`, `O3_transformations.jl` - Utilities

- **src/embed/** - Embedding layers (EdgeEmbed, EmbedDP), graph structures, spline transforms

- **src/transforms/** - Differentiation through NamedTuples, DecoratedParticles integration, Agnesi transforms

- **src/utils/** - Sparse product specs, inverse mappings, symmetry operations, type promotion

### Key Patterns

**Lux Integration**: Layers implement `LuxCore.AbstractLuxLayer` with `initialparameters` and `initialstates`.

**Allocation Management**: Uses Bumper.jl for fast allocations:
```julia
using WithAlloc: @withalloc
@withalloc evaluate!(layer, BB)
whatalloc(evaluate!, layer, args)  # Query allocation requirements
```

**Device Management**:
```julia
gpu_device(X)  # Move to GPU
cpu_device(X)  # Move to CPU
```

### Test Organization
Tests in `test/runtests.jl` are grouped by:
- Utils (SetProduct, InvMap)
- Embed (NamedTuples, Transform, DecoratedParticles, Splines)
- ACE Layers (StaticProd, SparseProdPool, SparseSymmetricProduct, SparseMatrix-KA)
- O3-Coupling (YYVector, Clebsch-Gordan, Representation, Coupling Coeffs)
- ACE Models (Pullback, ACE KA)
- Atoms (NeighbourListsExt)

Test utilities in `test/test_utils/` should be included before running individual tests.

## Dependencies

Key dependencies: Lux/LuxCore (neural networks), ChainRulesCore (autodiff), KernelAbstractions (GPU), ForwardDiff, PartialWaveFunctions (Clebsch-Gordan), StaticArrays, Polynomials4ML, DecoratedParticles, Bumper/WithAlloc.

Optional extensions: AtomsBase, NeighbourLists (in ext/).
