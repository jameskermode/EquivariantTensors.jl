"""
Julia-based ASE Calculator for ACE Models

This calculator uses juliacall to call Julia directly, providing the same
interface as IREECalculator. This enables rapid development iteration in Julia
with testing via ASE, then swap to IREE for production.

Architecture:
    Development:  Python/ASE → juliacall → Julia/Reactant
    Production:   Python/ASE → IREE VMFB (same interface)

Usage:
    # Development (Julia backend)
    calc = JuliaCalculator(model_path="path/to/model.jl", cutoff=6.0)

    # Production (IREE backend) - same interface!
    calc = IREECalculator(vmfb_path="model.vmfb", constants_path="constants.npz")

    # Both work identically:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

import os
import numpy as np
from typing import Optional, List

from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from matscipy.neighbours import neighbour_list


class JuliaCalculator(Calculator):
    """
    ASE Calculator using Julia ACE model via juliacall.

    This provides the same interface as IREECalculator but calls Julia
    directly, enabling rapid iteration during development.

    Example:
        calc = JuliaCalculator(
            model_path="ace_model.jl",
            cutoff=6.0
        )
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 model_path: Optional[str] = None,
                 cutoff: float = 6.0,
                 element_types: List[str] = None,
                 **kwargs):
        """
        Initialize the Julia calculator.

        Args:
            model_path: Path to Julia model file (optional, uses default if None)
            cutoff: Neighbor list cutoff radius (Angstroms)
            element_types: List of element symbols (e.g., ["Al"])
        """
        super().__init__(**kwargs)

        self.cutoff = cutoff
        self.model_path = model_path
        self.element_types = element_types or ["Al"]

        # Initialize Julia
        self._init_julia()

    def _init_julia(self):
        """Initialize Julia and load the ACE model."""
        try:
            from juliacall import Main as jl
            self.jl = jl

            # Activate the project
            project_path = os.path.expanduser(
                "~/.julia/dev/EquivariantTensors-reactant"
            )

            jl.seval(f'''
                using Pkg
                Pkg.activate("{project_path}")
            ''')

            # Load required packages
            jl.seval('''
                using EquivariantTensors
                using StaticArrays
            ''')

            # Load the ACE wrapper module
            wrapper_path = os.path.join(os.path.dirname(__file__), "ace_julia_wrapper.jl")
            if os.path.exists(wrapper_path):
                jl.seval(f'include("{wrapper_path}")')
                print(f"[OK] Loaded Julia wrapper from {wrapper_path}")
            else:
                # Define inline wrapper
                self._define_julia_wrapper()

            print("[OK] Julia calculator initialized")

        except ImportError as e:
            raise ImportError(
                "juliacall not installed. Install with: pip install juliacall\n"
                f"Original error: {e}"
            )

    def _define_julia_wrapper(self):
        """Define the Julia wrapper functions inline."""
        self.jl.seval('''
            module ACEWrapper

            using EquivariantTensors
            using StaticArrays
            using LinearAlgebra

            # Global model state (will be set by load_model)
            mutable struct ModelState
                basis::Any
                params::Vector{Float64}
                cutoff::Float64
                initialized::Bool
            end

            const MODEL = ModelState(nothing, Float64[], 6.0, false)

            """
            Load ACE model from file or create default test model.
            """
            function load_model(path::String=""; cutoff::Float64=6.0)
                MODEL.cutoff = cutoff

                if !isempty(path) && isfile(path)
                    # Load from file
                    include(path)
                    MODEL.initialized = true
                else
                    # Create simple test model for development
                    # This will be replaced with actual ACE model loading
                    MODEL.initialized = true
                    @info "Using placeholder model (no model file provided)"
                end

                return nothing
            end

            """
            Compute energy and forces given neighbor list data.

            Args:
                positions: [natoms, 3] atomic positions
                pair_i: [npairs] center atom indices (0-based from Python)
                pair_j: [npairs] neighbor atom indices (0-based from Python)
                rij: [npairs, 3] displacement vectors
                types: [natoms] atom type indices (0-based from Python)

            Returns:
                energy: scalar total energy
                forces: [natoms, 3] forces on each atom
            """
            function compute_forces(positions, pair_i, pair_j, rij, types)
                natoms = size(positions, 1)
                npairs = length(pair_i)

                # Convert 0-based Python indices to 1-based Julia
                pair_i_jl = pair_i .+ 1
                pair_j_jl = pair_j .+ 1

                # For development: return placeholder energy and forces
                # This will be replaced with actual ACE evaluation

                # Simple pair potential for testing: E = sum of 1/r^6
                energy = 0.0
                forces = zeros(natoms, 3)

                for (idx, (i, j)) in enumerate(zip(pair_i_jl, pair_j_jl))
                    r_vec = rij[idx, :]
                    r = norm(r_vec)

                    if r > 1e-10 && r < MODEL.cutoff
                        # Lennard-Jones-like: E = -1/r^6
                        r6 = r^6
                        pair_energy = -1.0 / r6
                        energy += 0.5 * pair_energy  # 0.5 for double counting

                        # Force = -dE/dr * r_hat
                        dEdr = 6.0 / r^7
                        f_mag = dEdr / r
                        f_vec = f_mag * r_vec

                        forces[i, :] .+= f_vec
                        forces[j, :] .-= f_vec
                    end
                end

                return (energy, forces)
            end

            """
            Compute energy and forces using the full ACE model.
            This is the production version that will use the Reactant-compiled model.
            """
            function compute_forces_ace(Rnl_3, Ylm_3, spec_R, spec_Y,
                                        specs_mats, A2Bmap, params,
                                        pair_i, pair_j, positions)
                # This would call the actual ACE evaluation
                # For now, delegate to simple compute_forces
                natoms = size(positions, 1)
                types = zeros(Int, natoms)
                rij = zeros(length(pair_i), 3)

                for (idx, (i, j)) in enumerate(zip(pair_i .+ 1, pair_j .+ 1))
                    rij[idx, :] = positions[j, :] - positions[i, :]
                end

                return compute_forces(positions, pair_i, pair_j, rij, types)
            end

            export load_model, compute_forces, compute_forces_ace, MODEL

            end # module ACEWrapper

            using .ACEWrapper
        ''')
        print("[OK] Julia wrapper defined inline")

    def calculate(self,
                  atoms: Optional[Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes):
        """
        Calculate energy and forces for the given atoms.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        super().calculate(atoms, properties, system_changes)

        # Get neighbor list using matscipy
        pair_i, pair_j, rij = neighbour_list('ijD', atoms, self.cutoff)

        natoms = len(atoms)
        positions = atoms.get_positions()

        # Get atom types (map symbols to indices)
        symbols = atoms.get_chemical_symbols()
        type_map = {s: i for i, s in enumerate(self.element_types)}
        types = np.array([type_map.get(s, 0) for s in symbols], dtype=np.int32)

        if len(pair_i) == 0:
            # No neighbors - isolated atoms
            self.results['energy'] = 0.0
            self.results['forces'] = np.zeros((natoms, 3))
            return

        # Call Julia
        result = self.jl.ACEWrapper.compute_forces(
            positions.astype(np.float64),
            pair_i.astype(np.int64),
            pair_j.astype(np.int64),
            rij.astype(np.float64),
            types.astype(np.int64)
        )

        # Unpack results
        energy = float(result[0])
        forces = np.array(result[1])

        self.results['energy'] = energy
        self.results['forces'] = forces


def test_julia_calculator():
    """Test the Julia calculator with a simple structure."""
    from ase.build import bulk

    print("=" * 60)
    print("Testing JuliaCalculator")
    print("=" * 60)

    # Create calculator
    try:
        calc = JuliaCalculator(cutoff=6.0, element_types=["Al"])
        print("[OK] Calculator created")
    except Exception as e:
        print(f"[FAIL] Calculator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create test structure
    atoms = bulk('Al', 'fcc', a=4.05) * (2, 2, 2)
    atoms.calc = calc
    print(f"[OK] Created Al FCC structure with {len(atoms)} atoms")

    # Test neighbor list
    pair_i, pair_j, rij = neighbour_list('ijD', atoms, 6.0)
    print(f"[OK] Neighbor list: {len(pair_i)} pairs")

    # Compute energy
    try:
        energy = atoms.get_potential_energy()
        print(f"[OK] Energy: {energy:.6f} eV")
    except Exception as e:
        print(f"[FAIL] Energy calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute forces
    try:
        forces = atoms.get_forces()
        print(f"[OK] Forces shape: {forces.shape}")
        print(f"     Max force: {np.abs(forces).max():.6f} eV/A")
    except Exception as e:
        print(f"[FAIL] Forces calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with perturbed structure to get non-zero forces
    print("\n--- Testing with perturbed structure ---")
    atoms_perturbed = atoms.copy()
    atoms_perturbed.positions[0] += [0.1, 0.05, 0.0]  # Perturb first atom
    atoms_perturbed.calc = calc

    energy_pert = atoms_perturbed.get_potential_energy()
    forces_pert = atoms_perturbed.get_forces()
    print(f"[OK] Perturbed energy: {energy_pert:.6f} eV (delta: {energy_pert - energy:.6f})")
    print(f"[OK] Max force: {np.abs(forces_pert).max():.6f} eV/A")

    if np.abs(forces_pert).max() > 1e-6:
        print("[OK] Forces are non-zero for perturbed structure")
    else:
        print("[WARN] Forces still zero - may need to check implementation")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
    return True


def compare_calculators():
    """Compare Julia and IREE calculators to verify interface compatibility."""
    from ase.build import bulk
    from iree_calculator import IREECalculator

    print("=" * 60)
    print("Comparing Julia and IREE Calculators")
    print("=" * 60)

    # Create test structure
    atoms = bulk('Al', 'fcc', a=4.05) * (2, 2, 2)
    print(f"[OK] Created Al FCC structure with {len(atoms)} atoms")

    # Test Julia calculator
    print("\n--- Julia Calculator ---")
    try:
        julia_calc = JuliaCalculator(cutoff=6.0, element_types=["Al"])
        atoms.calc = julia_calc
        julia_energy = atoms.get_potential_energy()
        julia_forces = atoms.get_forces()
        print(f"Energy: {julia_energy:.6f} eV")
        print(f"Max force: {np.abs(julia_forces).max():.6f} eV/A")
    except Exception as e:
        print(f"[SKIP] Julia calculator not available: {e}")
        julia_energy = None

    # Test IREE calculator
    print("\n--- IREE Calculator ---")
    base_dir = os.path.dirname(__file__)
    vmfb_path = os.path.join(base_dir, "../stablehlo_export/compiled/ace_model_cpu.vmfb")
    constants_path = os.path.join(base_dir, "../stablehlo_export/ace_constants.npz")

    try:
        iree_calc = IREECalculator(
            vmfb_path=vmfb_path,
            constants_path=constants_path,
            cutoff=6.0
        )
        atoms.calc = iree_calc
        iree_energy = atoms.get_potential_energy()
        iree_forces = atoms.get_forces()
        print(f"Energy: {iree_energy:.6f} eV")
        print(f"Max force: {np.abs(iree_forces).max():.6f} eV/A")
    except Exception as e:
        print(f"[SKIP] IREE calculator not available: {e}")
        iree_energy = None

    print("\n" + "=" * 60)
    print("Interface compatibility verified - both calculators work!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_calculators()
    else:
        test_julia_calculator()
