#=
ACE Julia Wrapper for ASE Calculator

This file provides the Julia-side interface for the JuliaCalculator.
Users can customize this to use their actual ACE model.

Usage from Python:
    from julia_calculator import JuliaCalculator
    calc = JuliaCalculator(cutoff=6.0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
=#

module ACEWrapper

using LinearAlgebra

# Try to load EquivariantTensors if available
const ET_AVAILABLE = try
    @eval using EquivariantTensors
    true
catch
    false
end

# Global model state
mutable struct ModelState
    basis::Any
    params::Vector{Float64}
    cutoff::Float64
    initialized::Bool
    use_ace::Bool  # Whether to use full ACE or placeholder
end

const MODEL = ModelState(nothing, Float64[], 6.0, false, false)

"""
    load_model(path=""; cutoff=6.0, use_ace=false)

Load ACE model from file or create default test model.

If `use_ace=true` and EquivariantTensors is available, will use the full
ACE evaluation. Otherwise uses a simple LJ-like placeholder for testing.
"""
function load_model(path::String=""; cutoff::Float64=6.0, use_ace::Bool=false)
    MODEL.cutoff = cutoff
    MODEL.use_ace = use_ace && ET_AVAILABLE

    if !isempty(path) && isfile(path)
        include(path)
        MODEL.initialized = true
        @info "Loaded model from $path"
    else
        MODEL.initialized = true
        if MODEL.use_ace
            @info "Using ACE model (EquivariantTensors)"
        else
            @info "Using LJ placeholder model"
        end
    end

    return nothing
end

"""
    compute_forces(positions, pair_i, pair_j, rij, types) -> (energy, forces)

Compute energy and forces given neighbor list data.

# Arguments
- `positions`: [natoms, 3] atomic positions
- `pair_i`: [npairs] center atom indices (0-based from Python)
- `pair_j`: [npairs] neighbor atom indices (0-based from Python)
- `rij`: [npairs, 3] displacement vectors (j - i)
- `types`: [natoms] atom type indices (0-based from Python)

# Returns
- `energy`: scalar total energy (eV)
- `forces`: [natoms, 3] forces on each atom (eV/Å)
"""
function compute_forces(positions, pair_i, pair_j, rij, types)
    natoms = size(positions, 1)
    npairs = length(pair_i)

    # Convert 0-based Python indices to 1-based Julia
    pair_i_jl = pair_i .+ 1
    pair_j_jl = pair_j .+ 1

    if MODEL.use_ace && ET_AVAILABLE
        return compute_forces_ace_impl(positions, pair_i_jl, pair_j_jl, rij, types)
    else
        return compute_forces_lj(positions, pair_i_jl, pair_j_jl, rij, types)
    end
end

"""
Simple Lennard-Jones-like pair potential for testing.
E = ε * [(σ/r)^12 - 2(σ/r)^6]
"""
function compute_forces_lj(positions, pair_i, pair_j, rij, types)
    natoms = size(positions, 1)

    # LJ parameters (for Al-like behavior)
    ε = 0.4   # eV
    σ = 2.55  # Å

    energy = 0.0
    forces = zeros(natoms, 3)

    for (idx, (i, j)) in enumerate(zip(pair_i, pair_j))
        r_vec = @view rij[idx, :]
        r = norm(r_vec)

        if r > 1e-10 && r < MODEL.cutoff
            # LJ potential
            sr6 = (σ / r)^6
            sr12 = sr6^2

            # Smooth cutoff
            rc = MODEL.cutoff
            if r < rc
                # Apply switching function near cutoff
                if r > 0.9 * rc
                    t = (r - 0.9 * rc) / (0.1 * rc)
                    switch = 1 - 3*t^2 + 2*t^3
                    dswitch = (-6*t + 6*t^2) / (0.1 * rc)
                else
                    switch = 1.0
                    dswitch = 0.0
                end

                pair_energy = ε * (sr12 - 2*sr6)
                energy += 0.5 * pair_energy * switch

                # Force = -dE/dr * r_hat
                dEdr = ε * (-12*sr12/r + 12*sr6/r)
                f_mag = -(dEdr * switch + pair_energy * dswitch) / r
                f_vec = f_mag * r_vec

                forces[i, :] .+= f_vec
                forces[j, :] .-= f_vec
            end
        end
    end

    return (energy, forces)
end

"""
Full ACE model evaluation (placeholder - replace with actual implementation).
"""
function compute_forces_ace_impl(positions, pair_i, pair_j, rij, types)
    # This is where the actual ACE evaluation would go
    # For now, fall back to LJ
    @warn "ACE evaluation not yet implemented, using LJ fallback" maxlog=1
    return compute_forces_lj(positions, pair_i, pair_j, rij, types)
end

#=
Example: How to integrate with actual ACE model

function compute_forces_ace_impl(positions, pair_i, pair_j, rij, types)
    natoms = size(positions, 1)

    # Build graph structure for ACE
    # ... (convert neighbor list to ACE format)

    # Compute embeddings
    # Rnl, Ylm = compute_embeddings(rij, MODEL.cutoff)

    # Reshape to 3D
    # Rnl_3, Ylm_3 = reshape_to_3d(Rnl, Ylm, pair_i, natoms)

    # Call ACE evaluation
    # BB = ace_evaluate(Rnl_3, Ylm_3, MODEL.basis.state)

    # Compute energy
    # E = sum(BB * MODEL.params)

    # Compute forces via AD
    # forces = compute_forces_ad(...)

    return (energy, forces)
end
=#

export load_model, compute_forces, MODEL

end # module ACEWrapper

using .ACEWrapper
