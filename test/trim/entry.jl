# juliac --trim=safe entry point exercising the trim-safe construction API.
# test/trim/build_and_run.jl compiles this file and compares the executable's
# output with checksum_lines() evaluated by ordinary Julia.
import EquivariantTensors as ET
using SparseArrays: findnz
include(joinpath(@__DIR__, "..", "test_utils", "trim_cases.jl"))

_mix(h::UInt64, x::UInt64) = (h ⊻ x) * 0x100000001b3

function checksum_lines()::Vector{String}
   out = String[]
   for (maxn, maxl, ord) in TRIM_CASES, solver in (:sparse, :dense)
      mb = trim_mb_spec(maxn, maxl, ord); R = trim_rnl(maxn, maxl); Y = trim_ylm(maxl)
      t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real,
                                            nullspace_solver = solver)
      I, J, V = findnz(t.symm)
      h = 0xcbf29ce484222325
      for k in eachindex(V)
         h = _mix(h, UInt64(I[k])); h = _mix(h, UInt64(J[k])); h = _mix(h, reinterpret(UInt64, V[k]))
      end
      for (r, y) in t.Aspec_raw
         h = _mix(h, UInt64(r)); h = _mix(h, UInt64(y))
      end
      for bb in t.𝔸spec, b in bb
         h = _mix(h, UInt64(b.n)); h = _mix(h, UInt64(b.l)); h = _mix(h, UInt64(b.m + 1024))
      end
      push!(out, string(size(t.symm, 1), " ", size(t.symm, 2), " ", length(V), " ", h))
   end
   # L > 0 and both reflection symmetries (L is a compile-time constant: explicit calls)
   for rs in TRIM_REFL
      push!(out, _line_L(Val(0), rs)); push!(out, _line_L(Val(1), rs)); push!(out, _line_L(Val(2), rs))
      push!(out, _line_L(Val(3), rs)); push!(out, _line_L(Val(4), rs)); push!(out, _line_L(Val(5), rs))
      push!(out, _line_L(Val(6), rs))
   end
   return out
end

_hashval(h::UInt64, v::Float64) = _mix(h, reinterpret(UInt64, v))
function _hashval(h::UInt64, v::AbstractVector{Float64})
   for x in v
      h = _mix(h, reinterpret(UInt64, x))
   end
   return h
end

function _line_L(::Val{L}, rs::Symbol)::String where {L}
   maxn, maxl, ord = TRIM_L_CASE
   mb = trim_mb_spec(maxn, maxl, ord; all_parities = true)
   symm, 𝔸spec = ET.symmetrisation_matrix(Val(L), mb; prune = true, PI = true, basis = real,
                                         nullspace_solver = :dense, refl_sym = rs)
   I, J, V = findnz(symm)
   h = 0xcbf29ce484222325
   for k in eachindex(V)
      h = _mix(h, UInt64(I[k])); h = _mix(h, UInt64(J[k])); h = _hashval(h, V[k])
   end
   for bb in 𝔸spec, b in bb
      h = _mix(h, UInt64(b.n)); h = _mix(h, UInt64(b.l)); h = _mix(h, UInt64(b.m + 1024))
   end
   return string(L, " ", String(rs), " ", size(symm, 1)) * string(" ", size(symm, 2), " ", length(V), " ", h)
end

function (@main)(args::Vector{String})::Cint
   for line in checksum_lines()
      println(Core.stdout, line)
   end
   return 0
end
