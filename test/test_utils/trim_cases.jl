# Small coupling cases shared by test_trim_api.jl and test/trim/entry.jl.
const TRIM_NL = @NamedTuple{n::Int, l::Int}
const TRIM_LM = @NamedTuple{l::Int, m::Int}

# (maxn, maxl, max correlation order); (1, 2, 4) has degenerate nnll blocks,
# (1, 0, 8) reaches the table bound.
const TRIM_CASES = [(2, 2, 2), (2, 2, 3), (1, 2, 4), (1, 1, 5), (1, 0, 8)]

"""All sorted bodies of (n,l) channels (n ≤ maxn, l ≤ maxl) of length 1..ord with even ∑l."""
function trim_mb_spec(maxn::Int, maxl::Int, ord::Int; all_parities::Bool = false)
   chans = [(n = n, l = l) for l in 0:maxl for n in 1:maxn]
   mb = Vector{TRIM_NL}[]
   _trim_rec!(mb, TRIM_NL[], chans, ord, 1, all_parities)
   return mb
end

# (a top-level recursive function, not a closure: a recursive closure is boxed,
# which --trim cannot resolve)
function _trim_rec!(mb::Vector{Vector{TRIM_NL}}, bb::Vector{TRIM_NL}, chans::Vector{TRIM_NL},
                    ord::Int, start::Int, all_parities::Bool = false)
   if !isempty(bb) && (all_parities || iseven(sum(b.l for b in bb)))
      push!(mb, copy(bb))
   end
   length(bb) == ord && return
   for i in start:length(chans)
      push!(bb, chans[i]); _trim_rec!(mb, bb, chans, ord, i, all_parities); pop!(bb)
   end
   return
end

trim_rnl(maxn::Int, maxl::Int) = TRIM_NL[(n = n, l = l) for l in 0:maxl for n in 1:maxn]
trim_ylm(maxl::Int) = TRIM_LM[(l = l, m = m) for l in 0:maxl for m in -l:l]

# L > 0 and both reflection symmetries: bodies of either ∑l parity (all_parities = true)
const TRIM_L_CASE = (2, 2, 3)
const TRIM_REFL = (:sym, :asym)
