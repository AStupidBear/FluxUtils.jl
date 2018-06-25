using Flux: RNNCell, LSTMCell, GRUCell, Recur
using Flux: children
using DataFlow: OSet

namedchildren(x) = [(:nothing, c) for c in children(x)]

namedchildren(m::Union{Dense, Flux.Diagonal, LayerNorm, Conv, RNNCell, LSTMCell, GRUCell}) = zip(fieldnames(m), children(m))
    
namedchildren(m::Recur) = zip((:cell, :init), children(m))

namedchildren(c::Chain) = [(Symbol(typename(l)), l) for l in c.layers]

namedchildren(BN::BatchNorm) = zip((:λ, :β, :γ, :μ, :σ, :ϵ, :momentum, :active),
                    (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum, BN.active))


function namedprefor(f, namedx; seen = OSet())
    name, x = namedx
    x ∈ seen && return
    f(name, x)
    foreach(namedy -> namedprefor(f, namedy, seen = seen), namedchildren(x))
    return
end

function namedparams(m)
  ps = Any[]
  namedprefor((name, p) ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
        !any(p′ -> p′[2] === p, ps) && push!(ps, (name, p)),
    (Symbol(typename(m)), m))
  return ps
end
