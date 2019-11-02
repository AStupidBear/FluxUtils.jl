export namedparams, states, loadstates!, weights, namedweights

function typename(x::T) where T
    name, ext = splitext(string(T.name))
    isempty(ext) ? name : ext[2:end]
end

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
        istracked(p) && isleaf(p) &&
        !any(p′ -> p′[2] === p, ps) && push!(ps, (name, p)),
    (Symbol(typename(m)), m))
    return ps
end

function states(m)
    ss = Any[]
    prefor(m) do x
      x isa Recur || return 
      x.state isa Tuple ? push!(ss, x.state...) : push!(ss, x.state)
    end
    return ss
end

function loadstates!(m, xs)
    for (s, x) in zip(states(m), xs)
    size(s) == size(x) ||
        error("Expected param size $(size(s)), got $(size(x))")
    copyto!(data(s), data(x))
    end
end

function weights(m)
    ws, seen = [], IdSet()
    prefor(m, seen = seen) do w
        w isa AbstractArray || return
        push!(seen, w)
        push!(ws, w)
    end
    return ws
end

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
