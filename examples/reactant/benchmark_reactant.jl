#=
Reactant @compile Benchmark: CPU vs GPU
=======================================

Run with: julia --project benchmark_reactant.jl
=#

using Printf
using LinearAlgebra, StaticArrays
using SpheriCart
using BenchmarkTools
using Reactant

## ============================================================================
## Self-Contained Reactant-Compatible Functions
## ============================================================================

_sizeY(maxL) = (maxL + 1)^2
_lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

function evaluate_reactant_cheb(N::Int, x::AbstractVector{T}) where {T}
    P = similar(x, T, length(x), N)
    P[:, 1] .= one(T)
    if N > 1; P[:, 2] .= x; end
    for k = 3:N
        @views P[:, k] .= 2 .* x .* P[:, k-1] .- P[:, k-2]
    end
    return P
end

function compute_reactant_ylm(L::Int, Flm, x::AbstractVector{T},
                               y::AbstractVector{T}, z::AbstractVector{T}) where {T}
    nX = length(x)
    len = _sizeY(L)
    Z = similar(x, T, nX, len)
    rt2 = sqrt(T(2))

    r² = x .* x .+ y .* y .+ z .* z
    s = similar(Z, nX, L+1)
    c = similar(Z, nX, L+1)
    s[:, 1] .= zero(T); c[:, 1] .= one(T)
    for m = 1:L
        @views s[:, m+1] .= s[:, m] .* x .+ c[:, m] .* y
        @views c[:, m+1] .= c[:, m] .* x .- s[:, m] .* y
    end
    Q = similar(Z, nX, len)
    i00 = _lm2idx(0, 0)
    Q[:, i00] .= one(T)
    Z[:, i00] .= (Flm[1,1]/rt2) .* Q[:, i00]
    c[:, 1] .= one(T)/rt2

    for l = 1:L
        ill = _lm2idx(l, l); il⁻l = _lm2idx(l, -l)
        ill⁻¹ = _lm2idx(l, l-1); il⁻¹l⁻¹ = _lm2idx(l-1, l-1)
        il⁻l⁺¹ = _lm2idx(l, -l+1)
        F_l_l = Flm[1+l,1+l]; F_l_l⁻¹ = Flm[1+l,1+l-1]
        @views Q[:, ill] .= -(2*l-1) .* Q[:, il⁻¹l⁻¹]
        @views Z[:, ill] .= F_l_l .* Q[:, ill] .* c[:, l+1]
        @views Z[:, il⁻l] .= F_l_l .* Q[:, ill] .* s[:, l+1]
        @views Q[:, ill⁻¹] .= (2*l-1) .* z .* Q[:, il⁻¹l⁻¹]
        @views Z[:, il⁻l⁺¹] .= F_l_l⁻¹ .* Q[:, ill⁻¹] .* s[:, l]
        @views Z[:, ill⁻¹] .= F_l_l⁻¹ .* Q[:, ill⁻¹] .* c[:, l]
        for m = l-2:-1:0
            ilm = _lm2idx(l, m); il⁻m = _lm2idx(l, -m)
            il⁻¹m = _lm2idx(l-1, m); il⁻²m = _lm2idx(l-2, m)
            F_l_m = Flm[1+l,1+m]
            @views Q[:, ilm] .= ((2*l-1) .* z .* Q[:, il⁻¹m] .- (l+m-1) .* r² .* Q[:, il⁻²m]) ./ (l-m)
            @views Z[:, il⁻m] .= F_l_m .* Q[:, ilm] .* s[:, m+1]
            @views Z[:, ilm] .= F_l_m .* Q[:, ilm] .* c[:, m+1]
        end
    end
    return Z
end

function compute_radial_embedding(x, y, z, N::Int, rcut)
    T = eltype(x)
    r_sq = x .* x .+ y .* y .+ z .* z
    y_trans = one(T) ./ (one(T) .+ r_sq)
    ycut = one(T) / (one(T) + T(rcut)^2)
    env = (y_trans .- ycut).^2 .* (y_trans .+ ycut).^2
    return evaluate_reactant_cheb(N, y_trans) .* env
end

# Combined energy function for benchmarking
function combined_energy(x, y, z, N_cheb, rcut, maxl, Flm)
    Rnl = compute_radial_embedding(x, y, z, N_cheb, rcut)
    Ylm = compute_reactant_ylm(maxl, Flm, x, y, z)
    # Simple "energy" combining radial and angular
    n_common = min(size(Rnl, 2), size(Ylm, 2))
    return sum(Rnl[:, 1:n_common] .* Ylm[:, 1:n_common])
end

## ============================================================================
## Main Benchmark
## ============================================================================

function main()
    println("="^70)
    println("REACTANT @COMPILE BENCHMARK (Embeddings)")
    println("="^70)

    N_cheb, maxl, rcut = 10, 5, 5.0f0
    nedges = 4000

    edge_x = randn(Float32, nedges)
    edge_y = randn(Float32, nedges)
    edge_z = randn(Float32, nedges)

    ybasis = SpheriCart.SolidHarmonics(maxl; static=false, T=Float32)
    Flm = ybasis.Flm

    @printf("Test data: %d edges, N_cheb=%d, maxl=%d\n\n", nedges, N_cheb, maxl)

    # Julia baseline
    println("-"^70)
    println("Julia Baseline")
    println("-"^70)

    # Warmup
    Rnl_julia = compute_radial_embedding(edge_x, edge_y, edge_z, N_cheb, rcut)
    Ylm_julia = compute_reactant_ylm(maxl, Flm, edge_x, edge_y, edge_z)
    E_julia = combined_energy(edge_x, edge_y, edge_z, N_cheb, rcut, maxl, Flm)
    @printf("Combined energy: %.6f\n", E_julia)

    b1 = @benchmark compute_radial_embedding($edge_x, $edge_y, $edge_z, $N_cheb, $rcut) samples=50
    b2 = @benchmark compute_reactant_ylm($maxl, $Flm, $edge_x, $edge_y, $edge_z) samples=50
    b3 = @benchmark combined_energy($edge_x, $edge_y, $edge_z, $N_cheb, $rcut, $maxl, $Flm) samples=50

    t1_j, t2_j, t3_j = median(b1.times)/1e6, median(b2.times)/1e6, median(b3.times)/1e6
    @printf("  Radial embedding:   %.3f ms\n", t1_j)
    @printf("  Angular embedding:  %.3f ms\n", t2_j)
    @printf("  Combined energy:    %.3f ms\n", t3_j)

    # Reactant CPU
    println("\n" * "-"^70)
    println("Reactant CPU Compilation")
    println("-"^70)

    println("Compiling...")
    Reactant.set_default_backend("cpu")
    x_cpu = Reactant.to_rarray(edge_x)
    y_cpu = Reactant.to_rarray(edge_y)
    z_cpu = Reactant.to_rarray(edge_z)

    radial_cpu = Reactant.@compile compute_radial_embedding(x_cpu, y_cpu, z_cpu, N_cheb, rcut)
    ylm_cpu = Reactant.@compile compute_reactant_ylm(maxl, Flm, x_cpu, y_cpu, z_cpu)
    energy_cpu = Reactant.@compile combined_energy(x_cpu, y_cpu, z_cpu, N_cheb, rcut, maxl, Flm)

    # Verify correctness
    E_cpu = energy_cpu(x_cpu, y_cpu, z_cpu, N_cheb, rcut, maxl, Flm)
    @printf("Combined energy: %.6f (diff from Julia: %.2e)\n", Float64(E_cpu), abs(Float64(E_cpu) - E_julia))

    b4 = @benchmark $(radial_cpu)($x_cpu, $y_cpu, $z_cpu, $N_cheb, $rcut) samples=50
    b5 = @benchmark $(ylm_cpu)($maxl, $Flm, $x_cpu, $y_cpu, $z_cpu) samples=50
    b6 = @benchmark $(energy_cpu)($x_cpu, $y_cpu, $z_cpu, $N_cheb, $rcut, $maxl, $Flm) samples=50

    t1_c, t2_c, t3_c = median(b4.times)/1e6, median(b5.times)/1e6, median(b6.times)/1e6
    @printf("  Radial embedding:   %.3f ms (%.2fx vs Julia)\n", t1_c, t1_j/t1_c)
    @printf("  Angular embedding:  %.3f ms (%.2fx vs Julia)\n", t2_c, t2_j/t2_c)
    @printf("  Combined energy:    %.3f ms (%.2fx vs Julia)\n", t3_c, t3_j/t3_c)

    # Reactant GPU
    println("\n" * "-"^70)
    println("Reactant GPU Compilation")
    println("-"^70)

    # Free CPU Reactant resources
    GC.gc()

    println("Compiling...")
    Reactant.set_default_backend("gpu")
    x_gpu = Reactant.to_rarray(edge_x)
    y_gpu = Reactant.to_rarray(edge_y)
    z_gpu = Reactant.to_rarray(edge_z)

    # Only compile combined energy to reduce GPU memory pressure
    energy_gpu = Reactant.@compile combined_energy(x_gpu, y_gpu, z_gpu, N_cheb, rcut, maxl, Flm)

    # Verify correctness
    E_gpu = energy_gpu(x_gpu, y_gpu, z_gpu, N_cheb, rcut, maxl, Flm)
    @printf("Combined energy: %.6f (diff from Julia: %.2e)\n", Float64(E_gpu), abs(Float64(E_gpu) - E_julia))

    # Warmup GPU
    for _ in 1:5
        energy_gpu(x_gpu, y_gpu, z_gpu, N_cheb, rcut, maxl, Flm)
    end

    # Use fewer samples to avoid GPU OOM
    b7 = @benchmark $(energy_gpu)($x_gpu, $y_gpu, $z_gpu, $N_cheb, $rcut, $maxl, $Flm) samples=10 evals=1

    t1_g, t2_g = 0.0, 0.0  # Not measured individually on GPU
    t3_g = median(b7.times)/1e6
    @printf("  Combined energy:    %.3f ms (%.2fx vs Julia)\n", t3_g, t3_j/t3_g)

    # Summary table
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    @printf("%-20s %12s %12s %12s\n", "Component", "Julia", "React CPU", "React GPU")
    println("-"^60)
    @printf("%-20s %10.3f ms %10.3f ms %12s\n", "Radial", t1_j, t1_c, "-")
    @printf("%-20s %10.3f ms %10.3f ms %12s\n", "Angular", t2_j, t2_c, "-")
    @printf("%-20s %10.3f ms %10.3f ms %10.3f ms\n", "Combined", t3_j, t3_c, t3_g)
    println("-"^60)
    @printf("%-20s %12s %10.2fx %10.2fx\n", "Speedup (combined)", "1.00x", t3_j/t3_c, t3_j/t3_g)
    println("="^70)
end

main()
