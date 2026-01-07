"""
ACE ASE Calculators

This module provides two ASE Calculator implementations for ACE models:

1. JuliaCalculator - Development backend using juliacall
   - Calls Julia directly for rapid iteration
   - Full force computation via AD
   - No compilation step required

2. IREECalculator - Production backend using IREE
   - Uses pre-compiled VMFB binary
   - No Julia runtime dependency
   - Portable across systems

Both share the same interface, enabling easy switching:

    # Development
    from ace_calculators import create_calculator
    calc = create_calculator(backend="julia", cutoff=6.0)

    # Production
    calc = create_calculator(backend="iree", vmfb_path="model.vmfb")

    # Same usage
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

from typing import Optional, List


def create_calculator(
    backend: str = "julia",
    cutoff: float = 6.0,
    element_types: Optional[List[str]] = None,
    vmfb_path: Optional[str] = None,
    constants_path: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "local-task",
    **kwargs
):
    """
    Create an ACE calculator with the specified backend.

    Args:
        backend: "julia" for development, "iree" for production
        cutoff: Neighbor list cutoff (Angstroms)
        element_types: List of element symbols (e.g., ["Al"])
        vmfb_path: Path to IREE VMFB file (required for iree backend)
        constants_path: Path to model constants NPZ (required for iree backend)
        model_path: Path to Julia model file (optional for julia backend)
        device: IREE device ("local-task" for CPU, "cuda" for GPU)

    Returns:
        ASE Calculator instance

    Example:
        # Development with Julia
        calc = create_calculator(backend="julia", cutoff=6.0)

        # Production with IREE
        calc = create_calculator(
            backend="iree",
            vmfb_path="ace_model.vmfb",
            constants_path="ace_constants.npz"
        )
    """
    element_types = element_types or ["Al"]

    if backend.lower() == "julia":
        from .julia_calculator import JuliaCalculator
        return JuliaCalculator(
            model_path=model_path,
            cutoff=cutoff,
            element_types=element_types,
            **kwargs
        )

    elif backend.lower() == "iree":
        from .iree_calculator import IREECalculator

        if vmfb_path is None or constants_path is None:
            raise ValueError(
                "IREE backend requires vmfb_path and constants_path"
            )

        return IREECalculator(
            vmfb_path=vmfb_path,
            constants_path=constants_path,
            cutoff=cutoff,
            device=device,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'julia' or 'iree'")


__all__ = ['create_calculator', 'JuliaCalculator', 'IREECalculator']
