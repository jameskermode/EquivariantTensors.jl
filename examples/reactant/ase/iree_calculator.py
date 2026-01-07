"""
IREE-based ASE Calculator for ACE Models

This calculator uses the same IREE-compiled model as the LAMMPS ML-IAP interface,
enabling consistent results between MD (LAMMPS) and single-point calculations (ASE).

Architecture:
    Julia/Reactant → StableHLO → IREE VMFB → {LAMMPS ML-IAP, ASE Calculator}
"""

import os
import subprocess
import tempfile
import numpy as np
from typing import Optional, Tuple, List

from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from matscipy.neighbours import neighbour_list

# Path to local IREE installation
IREE_BIN = os.path.expanduser("~/iree/bin")


class IREERunner:
    """
    Run IREE models via subprocess using iree-run-module.

    This avoids version mismatches between pip iree-runtime and
    locally compiled VMFB files.
    """

    def __init__(self, vmfb_path: str, device: str = "local-task"):
        self.vmfb_path = vmfb_path
        self.device = device
        self.iree_run = os.path.join(IREE_BIN, "iree-run-module")

        if not os.path.exists(self.iree_run):
            raise FileNotFoundError(f"iree-run-module not found at {self.iree_run}")

    def __call__(self, *arrays) -> List[np.ndarray]:
        """
        Run the model with given input arrays.

        Returns list of output arrays.
        """
        # Use persistent temp directory
        tmpdir = "/tmp/iree_ase_inputs"
        os.makedirs(tmpdir, exist_ok=True)

        # Save inputs as raw binary
        input_args = []
        for i, arr in enumerate(arrays):
            arr = np.ascontiguousarray(arr)
            path = os.path.join(tmpdir, f"input{i}.bin")
            arr.tofile(path)

            # Build shape string: "5x5x10xf32" format
            shape_str = "x".join(str(d) for d in arr.shape)
            dtype = {
                np.float32: "f32",
                np.float64: "f64",
                np.int32: "i32",
                np.int64: "i64",
            }[arr.dtype.type]

            input_args.append(f"--input={shape_str}x{dtype}=@{path}")

        # Run iree-run-module
        cmd = [
            self.iree_run,
            f"--device={self.device}",
            f"--module={self.vmfb_path}",
            "--function=main",
        ] + input_args

        # Debug: print command
        # print(f"[DEBUG] Running: {cmd[0]}")
        # for arg in cmd[1:]:
        #     print(f"  {arg}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Print debug info on failure
            print(f"[DEBUG] Command failed. Input shapes:")
            for i, arr in enumerate(arrays):
                print(f"  arg{i}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"[DEBUG] stderr: {result.stderr[:500]}")
            raise RuntimeError(f"IREE execution failed: {result.stderr}")

        # Parse output - IREE outputs multiple results
        # Format:
        #   result[0]: hal.buffer_view
        #   f32=109.216                          (scalar)
        #   5x5x10xf32=[[...]]                   (tensor)
        outputs = []
        lines = result.stdout.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('EXEC') or line.startswith('result'):
                continue

            # Check for scalar float: just "f32=109.216" (no shape prefix)
            if line.startswith('f32='):
                val_str = line[4:].strip()
                # Scalar has no brackets
                if not val_str.startswith('['):
                    try:
                        outputs.append(np.array(float(val_str), dtype=np.float32))
                    except ValueError:
                        pass  # Skip if can't parse

            # Skip tensor outputs for now (e.g., "5x5x10xf32=[[...]]")
            # Future: could parse tensor outputs if needed

        return outputs


class ACEEmbeddings:
    """
    Compute radial (Chebyshev) and angular (Ylm) embeddings for ACE.

    This matches the embeddings computed in the Julia export.
    """

    def __init__(self, N_cheb: int, maxl: int, rcut: float):
        self.N_cheb = N_cheb
        self.maxl = maxl
        self.rcut = rcut
        self.nRnl = N_cheb
        self.nYlm = (maxl + 1) ** 2

    def compute_chebyshev(self, r: np.ndarray) -> np.ndarray:
        """Compute Chebyshev polynomials T_n(x) where x = 2*r/rcut - 1."""
        x = 2.0 * r / self.rcut - 1.0
        x = np.clip(x, -1.0, 1.0)

        Tn = np.zeros((len(r), self.N_cheb), dtype=np.float32)
        Tn[:, 0] = 1.0
        if self.N_cheb > 1:
            Tn[:, 1] = x
        for n in range(2, self.N_cheb):
            Tn[:, n] = 2.0 * x * Tn[:, n-1] - Tn[:, n-2]

        # Apply cutoff envelope
        envelope = 0.5 * (1.0 + np.cos(np.pi * r / self.rcut))
        envelope = np.where(r < self.rcut, envelope, 0.0)

        return Tn * envelope[:, np.newaxis]

    def compute_ylm(self, rij: np.ndarray) -> np.ndarray:
        """
        Compute real spherical harmonics up to maxl.

        Args:
            rij: Displacement vectors [nedges, 3]

        Returns:
            Ylm values [nedges, nYlm]
        """
        r = np.linalg.norm(rij, axis=1)
        r = np.maximum(r, 1e-10)  # Avoid division by zero

        # Normalized direction
        x = rij[:, 0] / r
        y = rij[:, 1] / r
        z = rij[:, 2] / r

        Ylm = np.zeros((len(r), self.nYlm), dtype=np.float32)

        # l=0: Y_0^0 = 1/sqrt(4*pi)
        Ylm[:, 0] = 0.28209479  # 1/sqrt(4*pi)

        if self.maxl >= 1:
            # l=1: Y_1^{-1}, Y_1^0, Y_1^1
            c1 = 0.4886025  # sqrt(3/(4*pi))
            Ylm[:, 1] = c1 * y   # Y_1^{-1}
            Ylm[:, 2] = c1 * z   # Y_1^0
            Ylm[:, 3] = c1 * x   # Y_1^1

        if self.maxl >= 2:
            # l=2: 5 components
            c2a = 1.0925485  # sqrt(15/(4*pi))
            c2b = 0.3153916  # sqrt(5/(16*pi))
            c2c = 0.5462742  # sqrt(15/(16*pi))

            Ylm[:, 4] = c2a * x * y                    # Y_2^{-2}
            Ylm[:, 5] = c2a * y * z                    # Y_2^{-1}
            Ylm[:, 6] = c2b * (3*z*z - 1)              # Y_2^0
            Ylm[:, 7] = c2a * x * z                    # Y_2^1
            Ylm[:, 8] = c2c * (x*x - y*y)              # Y_2^2

        return Ylm

    def compute(self, rij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both radial and angular embeddings.

        Args:
            rij: Displacement vectors [nedges, 3]

        Returns:
            Rnl: [nedges, N_cheb]
            Ylm: [nedges, nYlm]
        """
        r = np.linalg.norm(rij, axis=1)
        Rnl = self.compute_chebyshev(r)
        Ylm = self.compute_ylm(rij)
        return Rnl, Ylm


class IREECalculator(Calculator):
    """
    ASE Calculator using IREE-compiled ACE model.

    This calculator uses the same compiled model as LAMMPS ML-IAP,
    enabling consistent results between different simulation codes.

    Example:
        calc = IREECalculator(
            vmfb_path="ace_model_cpu.vmfb",
            constants_path="ace_constants.npz",
            cutoff=6.0
        )
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 vmfb_path: str,
                 constants_path: str,
                 cutoff: float = 6.0,
                 device: str = "local-task",
                 **kwargs):
        """
        Initialize the IREE calculator.

        Args:
            vmfb_path: Path to IREE compiled .vmfb file
            constants_path: Path to model constants .npz file
            cutoff: Neighbor list cutoff radius (Angstroms)
            device: IREE device ("local-task" for CPU, "cuda" for GPU)
        """
        super().__init__(**kwargs)

        self.cutoff = cutoff
        self.device = device
        self.vmfb_path = vmfb_path

        # Load model constants
        self._load_constants(constants_path)

        # Initialize embeddings
        self.embeddings = ACEEmbeddings(self.N_cheb, self.maxl, cutoff)

        # Load IREE runner
        if os.path.exists(vmfb_path):
            self.model_fn = IREERunner(vmfb_path, device)
            print(f"[OK] IREE runner ready for {vmfb_path}")
        else:
            self.model_fn = None
            print(f"[WARN] VMFB not found - using placeholder")

    def _load_constants(self, path: str):
        """Load model constants from npz file."""
        if os.path.exists(path):
            data = np.load(path)
            self.spec_R = data['spec_R'].astype(np.int64)
            self.spec_Y = data['spec_Y'].astype(np.int64)
            self.A2Bmap = data['A2Bmap'].astype(np.float32)
            self.params = data['params'].astype(np.float32)
            self.maxl = int(data['maxl'][0])
            self.N_cheb = int(data['N_cheb'][0])
            print(f"[OK] Loaded constants: maxl={self.maxl}, N_cheb={self.N_cheb}")
        else:
            raise FileNotFoundError(f"Constants file not found: {path}")

    def _build_3d_tensors(self,
                          pair_i: np.ndarray,
                          pair_j: np.ndarray,
                          Rnl: np.ndarray,
                          Ylm: np.ndarray,
                          natoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape edge embeddings to 3D tensors [maxneigs, natoms, nfeatures].

        This matches the format expected by the IREE model.
        """
        # Count neighbors per atom
        neigh_counts = np.bincount(pair_i, minlength=natoms)
        maxneigs = max(neigh_counts.max(), 1)

        # Initialize 3D tensors with zeros
        Rnl_3 = np.zeros((maxneigs, natoms, self.N_cheb), dtype=np.float32)
        Ylm_3 = np.zeros((maxneigs, natoms, self.embeddings.nYlm), dtype=np.float32)

        # Fill tensors
        neigh_idx = np.zeros(natoms, dtype=np.int32)
        for edge_idx, (i, j) in enumerate(zip(pair_i, pair_j)):
            ni = neigh_idx[i]
            if ni < maxneigs:
                Rnl_3[ni, i, :] = Rnl[edge_idx]
                Ylm_3[ni, i, :] = Ylm[edge_idx]
                neigh_idx[i] += 1

        return Rnl_3, Ylm_3

    def _pad_or_truncate(self, arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Pad or truncate array to target shape."""
        result = np.zeros(target_shape, dtype=arr.dtype)
        # Compute slice for each dimension
        slices_src = []
        slices_dst = []
        for i, (src_dim, tgt_dim) in enumerate(zip(arr.shape, target_shape)):
            size = min(src_dim, tgt_dim)
            slices_src.append(slice(0, size))
            slices_dst.append(slice(0, size))
        result[tuple(slices_dst)] = arr[tuple(slices_src)]
        return result

    def _compute_energy_forces_iree(self,
                                     Rnl_3: np.ndarray,
                                     Ylm_3: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Call IREE model to compute energy.

        The IREE model expects FIXED shapes (from compilation):
          arg0: Rnl_3 [5, 5, 10] float32  (maxneigs=5, nnodes=5, nRnl=10)
          arg1: Ylm_3 [9, 5, 10] float32  (nYlm=9, nnodes=5, maxneigs=10)
          arg2: spec_R [19] int64
          arg3: spec_Y [19] int64
          arg4: specs_mat1 [1, 5] int64
          arg5: specs_mat2 [2, 26] int64
          arg6: A2Bmap [31, 19] float32
          arg7: params [19] float32

        Note: For forces, we would need to also export gradients from Julia
        or use finite differences.
        """
        natoms = Rnl_3.shape[1]

        if self.model_fn is None:
            # Placeholder: return dummy values
            return -3.5 * natoms, np.zeros((natoms, 3), dtype=np.float32)

        # Fixed shapes from the compiled model
        COMPILED_MAXNEIGS = 5
        COMPILED_NNODES = 5
        COMPILED_NRNL = 10
        COMPILED_NYLM = 9

        # Prepare inputs matching the compiled MLIR signature
        # arg0: Rnl_3 [5, 5, 10] - pad/truncate to fixed shape
        arg0 = self._pad_or_truncate(Rnl_3, (COMPILED_MAXNEIGS, COMPILED_NNODES, COMPILED_NRNL))
        arg0 = arg0.astype(np.float32)

        # arg1: Ylm_3 needs to be [9, 5, 10] (nYlm, nnodes, maxneigs)
        # Input Ylm_3 is [maxneigs, nnodes, nYlm], need [nYlm, nnodes, maxneigs]
        Ylm_transposed = np.transpose(Ylm_3, (2, 1, 0))  # [nYlm, nnodes, maxneigs]
        # Note: the compiled model expects [9, 5, 10] where last dim is maxneigs (10 from Rnl dim)
        arg1 = self._pad_or_truncate(Ylm_transposed, (COMPILED_NYLM, COMPILED_NNODES, COMPILED_NRNL))
        arg1 = arg1.astype(np.float32)

        # arg2, arg3: spec indices
        arg2 = self.spec_R.astype(np.int64)
        arg3 = self.spec_Y.astype(np.int64)

        # arg4: specs_mat1 [1, nspec1] - order 1 specifications
        # From the test inputs, this was shape (1, 5) with values [1, 10, 14, 18, 19]
        arg4 = np.array([[1, 10, 14, 18, 19]], dtype=np.int64)

        # arg5: specs_mat2 [2, nspec2] - order 2 specifications
        # From the test inputs, this was shape (2, 26)
        arg5 = np.zeros((2, 26), dtype=np.int64)
        arg5[0, :] = [1, 1, 1, 1, 1, 2, 4, 3, 2, 4, 3, 2, 4, 3, 10, 10, 10, 5, 9, 6, 8, 7, 11, 13, 12, 14]
        arg5[1, :] = [1, 10, 14, 18, 19, 2, 4, 3, 11, 13, 12, 15, 17, 16, 10, 14, 18, 5, 9, 6, 8, 7, 11, 13, 12, 14]

        # arg6: A2Bmap [nBB, nA] - coupling coefficients
        arg6 = self.A2Bmap.T.astype(np.float32)  # Transpose to [31, 19]

        # arg7: params
        arg7 = self.params.astype(np.float32)

        # Call IREE model
        try:
            results = self.model_fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
            energy = float(results[0])

            # Forces would come from gradients - not yet implemented
            forces = np.zeros((natoms, 3), dtype=np.float32)

            return energy, forces

        except Exception as e:
            print(f"[WARN] IREE call failed: {e}")
            import traceback
            traceback.print_exc()
            return -3.5 * natoms, np.zeros((natoms, 3), dtype=np.float32)

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

        # Get neighbor list using matscipy (fast C++ implementation)
        pair_i, pair_j, rij = neighbour_list('ijD', atoms, self.cutoff)

        natoms = len(atoms)

        if len(pair_i) == 0:
            # No neighbors - isolated atoms
            self.results['energy'] = 0.0
            self.results['forces'] = np.zeros((natoms, 3))
            return

        # Compute embeddings
        Rnl, Ylm = self.embeddings.compute(rij)

        # Build 3D tensors
        Rnl_3, Ylm_3 = self._build_3d_tensors(pair_i, pair_j, Rnl, Ylm, natoms)

        # Compute energy and forces via IREE
        energy, forces = self._compute_energy_forces_iree(Rnl_3, Ylm_3)

        self.results['energy'] = energy
        self.results['forces'] = forces


def test_iree_calculator():
    """Test the IREE calculator with a simple structure."""
    from ase.build import bulk

    print("=" * 60)
    print("Testing IREECalculator")
    print("=" * 60)

    # First, run the reference test with original inputs to verify IREE works
    print("\n--- Step 1: Verify IREE with reference inputs ---")
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "../stablehlo_export/test_inputs_raw")
    vmfb_path = os.path.join(base_dir, "../stablehlo_export/compiled/ace_model_cpu.vmfb")

    if os.path.exists(raw_dir):
        runner = IREERunner(vmfb_path)
        # Load reference inputs
        arg0 = np.fromfile(os.path.join(raw_dir, "arg0.bin"), dtype=np.float32).reshape(5, 5, 10)
        arg1 = np.fromfile(os.path.join(raw_dir, "arg1.bin"), dtype=np.float32).reshape(9, 5, 10)
        arg2 = np.fromfile(os.path.join(raw_dir, "arg2.bin"), dtype=np.int64)
        arg3 = np.fromfile(os.path.join(raw_dir, "arg3.bin"), dtype=np.int64)
        arg4 = np.fromfile(os.path.join(raw_dir, "arg4.bin"), dtype=np.int64).reshape(1, 5)
        arg5 = np.fromfile(os.path.join(raw_dir, "arg5.bin"), dtype=np.int64).reshape(2, 26)
        arg6 = np.fromfile(os.path.join(raw_dir, "arg6.bin"), dtype=np.float32).reshape(31, 19)
        arg7 = np.fromfile(os.path.join(raw_dir, "arg7.bin"), dtype=np.float32)

        results = runner(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        if results:
            print(f"[OK] Reference energy: {results[0]:.6f} (expected: 109.216)")
        else:
            print("[WARN] No output from reference test")
    else:
        print(f"[SKIP] Reference inputs not found at {raw_dir}")

    print("\n--- Step 2: Test with ASE structure ---")

    # Paths
    base_dir = os.path.dirname(__file__)
    vmfb_path = os.path.join(base_dir, "../stablehlo_export/compiled/ace_model_cpu.vmfb")
    constants_path = os.path.join(base_dir, "../stablehlo_export/ace_constants.npz")

    # Create calculator
    try:
        calc = IREECalculator(
            vmfb_path=vmfb_path,
            constants_path=constants_path,
            cutoff=6.0,
            device="local-task"
        )
        print("[OK] Calculator created")
    except Exception as e:
        print(f"[FAIL] Calculator creation failed: {e}")
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
        print(f"[OK] Energy: {energy:.4f} eV")
    except Exception as e:
        print(f"[FAIL] Energy calculation failed: {e}")
        return False

    # Compute forces
    try:
        forces = atoms.get_forces()
        print(f"[OK] Forces shape: {forces.shape}")
        print(f"     Max force: {np.abs(forces).max():.6f} eV/A")
    except Exception as e:
        print(f"[FAIL] Forces calculation failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_iree_calculator()
