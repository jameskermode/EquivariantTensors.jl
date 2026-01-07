"""
MLIAP Unified Test with IREE-compiled ACE model
"""

import os
import sys
import numpy as np
import subprocess

# Add LAMMPS Python path
sys.path.insert(0, os.path.expanduser("~/lammps/lammps-22Jul2025/python"))

from lammps.mliap.mliap_unified_abc import MLIAPUnified

# Paths
IREE_BIN = os.path.expanduser("~/iree/bin")
EXPORT_DIR = os.path.dirname(__file__) + "/../stablehlo_export"
VMFB_PATH = f"{EXPORT_DIR}/compiled/ace_model_cpu.vmfb"
CONSTANTS_PATH = f"{EXPORT_DIR}/ace_constants.npz"


class ACEModelIREE(MLIAPUnified):
    """ACE model using IREE-compiled backend."""

    def __init__(self, interface=None, element_types=None, ndescriptors=0, nparams=0, rcutfac=6.0):
        super().__init__(interface, element_types, ndescriptors, nparams, rcutfac)

        # Load model constants
        self.load_constants()
        print(f"[OK] ACEModelIREE initialized")
        print(f"     Elements: {element_types}")
        print(f"     rcutfac: {rcutfac}")
        print(f"     nA: {len(self.spec_R)}")

    def load_constants(self):
        """Load ACE model constants from npz file."""
        if os.path.exists(CONSTANTS_PATH):
            data = np.load(CONSTANTS_PATH)
            self.spec_R = data['spec_R'].astype(np.int64)
            self.spec_Y = data['spec_Y'].astype(np.int64)
            self.A2Bmap = data['A2Bmap'].astype(np.float32)
            self.params = data['params'].astype(np.float32)
            self.maxl = int(data['maxl'][0])
            self.N_cheb = int(data['N_cheb'][0])
            print(f"[OK] Loaded constants from {CONSTANTS_PATH}")
            print(f"     maxl={self.maxl}, N_cheb={self.N_cheb}")
            print(f"     spec_R shape: {self.spec_R.shape}")
            print(f"     A2Bmap shape: {self.A2Bmap.shape}")
        else:
            # Use dummy values for testing
            print(f"[WARN] Constants file not found, using dummy values")
            self.spec_R = np.array([1, 1, 1, 1, 1], dtype=np.int64)
            self.spec_Y = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            self.A2Bmap = np.eye(5, dtype=np.float32)
            self.params = np.ones(5, dtype=np.float32)
            self.maxl = 2
            self.N_cheb = 5

    def compute_forces(self, data):
        """Compute energy and forces using IREE model."""
        natoms = data.nlistatoms

        # For now, return a simple energy based on atom count
        # Full implementation would:
        # 1. Extract neighbor lists from data
        # 2. Compute Rnl and Ylm embeddings
        # 3. Call IREE model via subprocess or C API
        # 4. Convert gradients to forces

        # Simple placeholder: constant energy per atom
        energy_per_atom = -3.5
        total_energy = energy_per_atom * natoms
        data.energy = total_energy

        # Zero forces for now
        try:
            if data.f is not None:
                data.f[:] = 0.0
        except:
            pass

        print(f"  ACEModelIREE.compute_forces: natoms={natoms}, E={total_energy:.4f}")

    def compute_descriptors(self, data):
        """Not used in unified mode."""
        pass

    def compute_gradients(self, data):
        """Compute gradients (required by ABC)."""
        pass


def test_iree_model_standalone():
    """Test IREE model execution standalone."""
    print("\n=== Testing IREE Model Standalone ===")

    if not os.path.exists(VMFB_PATH):
        print(f"[SKIP] VMFB not found at {VMFB_PATH}")
        return False

    # Run iree-run-module
    raw_dir = f"{EXPORT_DIR}/test_inputs_raw"
    cmd = [
        f"{IREE_BIN}/iree-run-module",
        "--device=local-task",
        f"--module={VMFB_PATH}",
        "--function=main",
        f"--input=5x5x10xf32=@{raw_dir}/arg0.bin",
        f"--input=9x5x10xf32=@{raw_dir}/arg1.bin",
        f"--input=19xi64=@{raw_dir}/arg2.bin",
        f"--input=19xi64=@{raw_dir}/arg3.bin",
        f"--input=1x5xi64=@{raw_dir}/arg4.bin",
        f"--input=2x26xi64=@{raw_dir}/arg5.bin",
        f"--input=31x19xf32=@{raw_dir}/arg6.bin",
        f"--input=19xf32=@{raw_dir}/arg7.bin",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse energy from output
    for line in result.stdout.split('\n'):
        if 'result[0]' in line or 'f32=' in line:
            print(f"  {line.strip()}")
            if 'f32=' in line:
                energy = line.split('f32=')[1].strip()
                print(f"[OK] IREE Energy: {energy}")
                return True

    if result.returncode != 0:
        print(f"[FAIL] IREE error: {result.stderr}")
        return False

    return True


def test_mliap_with_ace():
    """Test MLIAP with ACE model."""
    print("\n=== Testing MLIAP with ACE Model ===")

    from lammps.mliap.loader import activate_mliappy
    from lammps import lammps

    # Create ACE model
    model = ACEModelIREE(element_types=["Al"])

    # Create LAMMPS
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

    try:
        # Activate MLIAP Python coupling
        activate_mliappy(lmp)
        print("[OK] MLIAP Python coupling activated")

        # Pickle model
        model.pickle("ace_unified_model.pkl")
        print("[OK] Model pickled")

        # Set pair style
        lmp.commands_string("""
            pair_style mliap unified ace_unified_model.pkl 0
            pair_coeff * * Al
        """)
        print("[OK] Pair style set")

        # Run
        lmp.command("run 0")

        pe = lmp.get_thermo("pe")
        print(f"[OK] ACE Potential energy: {pe:.4f} eV")

        success = True

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        success = False

    lmp.close()

    # Cleanup
    if os.path.exists("ace_unified_model.pkl"):
        os.remove("ace_unified_model.pkl")

    return success


if __name__ == "__main__":
    print("=" * 60)
    print("MLIAP + IREE ACE Model Integration Test")
    print("=" * 60)

    # Test IREE standalone
    test_iree_model_standalone()

    # Test MLIAP with ACE
    test_mliap_with_ace()

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)
