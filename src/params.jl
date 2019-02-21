using Flux: RNNCell, LSTMCell, GRUCell, Recur
using Flux: children
using Flux.Tracker: IdSet

export namedparams

namedchildren(x) = [(:nothing, c) for c in children(x)]

namedchildren(m::Union{Dense, Flux.Diagonal, LayerNorm, Conv, RNNCell, LSTMCell, GRUCell}) = zip(fieldnames(typeof(m)), children(m))
    
namedchildren(m::Recur) = zip((:cell, :init), children(m))

namedchildren(c::Chain) = [(Symbol(typename(l)), l) for l in c.layers]

namedchildren(BN::BatchNorm) = zip((:λ, :β, :γ, :μ, :σ, :ϵ, :momentum, :active),
                    (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum, BN.active))


function namedprefor(f, namedx; seen = IdSet())
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

export states
function states(m)
    ss = Any[]
    Flux.prefor(m) do x
      x isa Recur || return 
      x.state isa Tuple ? push!(ss, x.state...) : push!(ss, x.state)
    end
    return ss
end

export loadstates!
function loadstates!(m, xs)
    for (s, x) in zip(states(m), xs)
    size(s) == size(x) ||
        error("Expected param size $(size(s)), got $(size(x))")
    copyto!(Flux.data(s), Flux.data(x))
    end
end

export weights
function weights(m)
    ws, seen = [], IdSet()
    Flux.prefor(m, seen = seen) do w
        w isa AbstractArray || return
        push!(seen, w)
        push!(ws, w)
    end
    return ws
end

export namedweights
function namedweights(m)
    ws, seen = [], IdSet()
    mname = Symbol(typename(m))
    namedprefor((mname, m), seen = seen) do name, w
        w isa AbstractArray || return
        push!(seen, w)
        push!(ws, (name, w))
    end
    return ws
end
