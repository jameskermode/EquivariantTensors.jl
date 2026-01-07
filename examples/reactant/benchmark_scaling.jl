#=
Scaling Benchmark: Reactant GPU vs CPU across system sizes
==========================================================

Tests the hypothesis that larger systems benefit more from Reactant GPU acceleration.

Run with: julia --project benchmark_scaling.jl
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
    n_common = min(size(Rnl, 2), size(Ylm, 2))
    return sum(Rnl[:, 1:n_common] .* Ylm[:, 1:n_common])
end

## ============================================================================
## Benchmark for a single size
## ============================================================================

function benchmark_size(nedges::Int, N_cheb::Int, maxl::Int, rcut::Float32, Flm)
    # Generate test data
    edge_x = randn(Float32, nedges)
    edge_y = randn(Float32, nedges)
    edge_z = randn(Float32, nedges)

    # Julia baseline
    E_julia = combined_energy(edge_x, edge_y, edge_z, N_cheb, rcut, maxl, Flm)
    b_julia = @benchmark combined_energy($edge_x, $edge_y, $edge_z, $N_cheb, $rcut, $maxl, $Flm) samples=15 evals=1
    t_julia = median(b_julia.times) / 1e6

    # Reactant CPU
    Reactant.set_default_backend("cpu")
    x_cpu = Reactant.to_rarray(edge_x)
    y_cpu = Reactant.to_rarray(edge_y)
    z_cpu = Reactant.to_rarray(edge_z)

    energy_cpu = Reactant.@compile combined_energy(x_cpu, y_cpu, z_cpu, N_cheb, rcut, maxl, Flm)

    # Warmup
    for _ in 1:3
        energy_cpu(x_cpu, y_cpu, z_cpu, N_cheb, rcut, maxl, Flm)
    end

    b_cpu = @benchmark $(energy_cpu)($x_cpu, $y_cpu, $z_cpu, $N_cheb, $rcut, $maxl, $Flm) samples=15 evals=1
    t_cpu = median(b_cpu.times) / 1e6

    # Reactant GPU
    GC.gc()
    Reactant.set_default_backend("gpu")
    x_gpu = Reactant.to_rarray(edge_x)
    y_gpu = Reactant.to_rarray(edge_y)
    z_gpu = Reactant.to_rarray(edge_z)

    energy_gpu = Reactant.@compile combined_energy(x_gpu, y_gpu, z_gpu, N_cheb, rcut, maxl, Flm)

    # Warmup
    for _ in 1:5
        energy_gpu(x_gpu, y_gpu, z_gpu, N_cheb, rcut, maxl, Flm)
    end

    b_gpu = @benchmark $(energy_gpu)($x_gpu, $y_gpu, $z_gpu, $N_cheb, $rcut, $maxl, $Flm) samples=15 evals=1
    t_gpu = median(b_gpu.times) / 1e6

    # Cleanup
    GC.gc()

    return (t_julia=t_julia, t_cpu=t_cpu, t_gpu=t_gpu,
            speedup_cpu=t_julia/t_cpu, speedup_gpu=t_julia/t_gpu)
end

## ============================================================================
## Main Scaling Benchmark
## ============================================================================

function main()
    println("="^80)
    println("SCALING BENCHMARK: Reactant GPU vs CPU across system sizes")
    println("="^80)

    N_cheb, maxl, rcut = 10, 5, 5.0f0
    ybasis = SpheriCart.SolidHarmonics(maxl; static=false, T=Float32)
    Flm = ybasis.Flm

    @printf("\nModel parameters: N_cheb=%d, maxl=%d, rcut=%.1f\n\n", N_cheb, maxl, rcut)

    # Test sizes: from small to large
    # Typical MLIP systems: ~100 atoms * ~30 neighbors = ~3000 edges
    # Large systems: ~10000 atoms * ~30 neighbors = ~300000 edges
    sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

    println("-"^80)
    @printf("%-12s %12s %12s %12s %12s %12s\n",
            "Edges", "Julia (ms)", "CPU (ms)", "GPU (ms)", "CPU speedup", "GPU speedup")
    println("-"^80)

    results = []

    for nedges in sizes
        @printf("Testing %6d edges... ", nedges)
        flush(stdout)

        try
            r = benchmark_size(nedges, N_cheb, maxl, rcut, Flm)
            push!(results, (nedges=nedges, r...))

            @printf("\r%-12d %12.3f %12.3f %12.3f %12.2fx %12.2fx\n",
                    nedges, r.t_julia, r.t_cpu, r.t_gpu, r.speedup_cpu, r.speedup_gpu)
        catch e
            @printf("\r%-12d %s\n", nedges, "ERROR: $(typeof(e))")
            if nedges >= 64000
                println("  (GPU OOM likely - stopping at this size)")
                break
            end
        end

        # Force cleanup between sizes
        GC.gc()
    end

    # Summary
    println("\n" * "="^80)
    println("ANALYSIS")
    println("="^80)

    if length(results) >= 2
        small = results[1]
        large = results[end]

        println("\nSmallest system ($(small.nedges) edges):")
        @printf("  Julia: %.3f ms, CPU: %.3f ms (%.2fx), GPU: %.3f ms (%.2fx)\n",
                small.t_julia, small.t_cpu, small.speedup_cpu, small.t_gpu, small.speedup_gpu)

        println("\nLargest system ($(large.nedges) edges):")
        @printf("  Julia: %.3f ms, CPU: %.3f ms (%.2fx), GPU: %.3f ms (%.2fx)\n",
                large.t_julia, large.t_cpu, large.speedup_cpu, large.t_gpu, large.speedup_gpu)

        # Find crossover point where GPU beats CPU
        crossover = nothing
        for r in results
            if r.speedup_gpu > r.speedup_cpu
                crossover = r.nedges
                break
            end
        end

        if crossover !== nothing
            println("\nGPU surpasses CPU speedup at: ~$(crossover) edges")
        else
            println("\nGPU did not surpass CPU in tested range")
        end

        # Calculate scaling trends
        if length(results) >= 3
            # Linear regression on log-log scale for GPU speedup
            x = log.(Float64.([r.nedges for r in results]))
            y_gpu = log.(Float64.([r.speedup_gpu for r in results]))
            y_cpu = log.(Float64.([r.speedup_cpu for r in results]))

            # Simple linear fit: y = a + b*x
            n = length(x)
            b_gpu = (n * sum(x .* y_gpu) - sum(x) * sum(y_gpu)) / (n * sum(x.^2) - sum(x)^2)
            b_cpu = (n * sum(x .* y_cpu) - sum(x) * sum(y_cpu)) / (n * sum(x.^2) - sum(x)^2)

            println("\nScaling analysis (log-log slope):")
            @printf("  GPU speedup scales as: N^%.3f\n", b_gpu)
            @printf("  CPU speedup scales as: N^%.3f\n", b_cpu)

            if b_gpu > 0.1
                println("\n  => GPU speedup INCREASES with system size (as expected)")
            elseif b_gpu < -0.1
                println("\n  => GPU speedup decreases with system size")
            else
                println("\n  => GPU speedup roughly constant with system size")
            end
        end
    end

    println("\n" * "="^80)
end

main()
