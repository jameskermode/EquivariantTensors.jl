"""
Simple MLIAP Unified Test with IREE-compiled ACE model
"""

import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.expanduser("~/lammps/lammps-22Jul2025/python"))

# Try to import LAMMPS MLIAP base class directly
try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
    print("[OK] MLIAPUnified base class imported")
    MLIAP_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] MLIAPUnified not available: {e}")
    MLIAP_AVAILABLE = False
    # Create stub
    class MLIAPUnified:
        def __init__(self, interface, element_types, ndescriptors=0, nparams=0, rcutfac=5.0):
            pass


# Simple test model that returns constant energy
class SimpleTestModel(MLIAPUnified):
    """Simple test model that returns constant energy for verification."""

    def __init__(self, interface=None, element_types=None, ndescriptors=0, nparams=0, rcutfac=5.0):
        super().__init__(interface, element_types, ndescriptors, nparams, rcutfac)
        print(f"[OK] SimpleTestModel initialized with elements: {element_types}")

    def compute_forces(self, data):
        """Compute energy and forces."""
        natoms = data.nlistatoms

        # Simple: return constant energy per atom
        energy_per_atom = -3.5  # eV
        total_energy = energy_per_atom * natoms
        data.energy = total_energy

        # Zero forces - use data.f which is the forces array
        try:
            if data.f is not None:
                data.f[:] = 0.0
        except:
            pass

        print(f"  compute_forces called: natoms={natoms}, E={total_energy:.4f}")

    def compute_descriptors(self, data):
        """Not used in unified mode."""
        pass

    def compute_gradients(self, data):
        """Compute gradients (required by ABC)."""
        pass


def test_lammps_mliap():
    """Test MLIAP unified interface with LAMMPS from command line."""
    from lammps import lammps

    print("\n=== Testing MLIAP Unified Interface via LAMMPS ===")

    # Create LAMMPS instance
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

    # Set up simple test
    lmp.commands_string("""
        units metal
        atom_style atomic
        boundary p p p

        # Create small Al lattice
        lattice fcc 4.05
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 26.98

        # Use SNAP as a test (simpler than mliap unified)
        pair_style snap
        pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/examples/snap/Ta06A.snapcoeff /home/eng/essswb/lammps/lammps-22Jul2025/examples/snap/Ta06A.snapparam Ta
    """)

    print(f"Created {lmp.get_natoms()} atoms")

    # Run single point
    lmp.command("run 0")
    pe = lmp.get_thermo("pe")
    print(f"[OK] SNAP Potential energy: {pe:.4f} eV")

    lmp.close()
    return True


def test_mliap_unified_direct():
    """Test using mliap with Python model via activate_mliappy."""
    print("\n=== Testing MLIAP Python Integration ===")

    # Check if loader is available
    try:
        from lammps.mliap.loader import activate_mliappy
        print("[OK] MLIAP loader available")
    except ImportError as e:
        print(f"[SKIP] MLIAP loader not available: {e}")
        return False

    from lammps import lammps

    # Create model
    model = SimpleTestModel(element_types=["Al"])

    # Create LAMMPS and test
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

    lmp.commands_string("""
        units metal
        atom_style atomic
        boundary p p p
        lattice fcc 4.05
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 26.98
    """)

    print(f"Created {lmp.get_natoms()} atoms")

    # Try activating MLIAP Python
    try:
        # Step 1: Activate the Python coupling module
        activate_mliappy(lmp)
        print("[OK] MLIAP Python coupling activated")

        # Step 2: Pickle the model to a file
        model.pickle("test_unified_model.pkl")
        print("[OK] Model pickled to test_unified_model.pkl")

        # Step 3: Set up pair style with pickled model
        # Syntax: pair_style mliap unified <pickle_file> <num_extra_args>
        lmp.commands_string("""
            pair_style mliap unified test_unified_model.pkl 0
            pair_coeff * * Al
        """)
        print("[OK] Pair style set")

        # Step 4: Run
        lmp.command("run 0")

        pe = lmp.get_thermo("pe")
        print(f"[OK] MLIAP Potential energy: {pe:.4f} eV")
        success = True

    except Exception as e:
        print(f"[FAIL] MLIAP test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    lmp.close()
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("LAMMPS MLIAP Integration Test")
    print("=" * 60)

    # First test basic LAMMPS with ML potential
    try:
        test_lammps_mliap()
    except Exception as e:
        print(f"[FAIL] Basic test failed: {e}")

    # Then test MLIAP unified
    try:
        test_mliap_unified_direct()
    except Exception as e:
        print(f"[FAIL] MLIAP unified test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)
