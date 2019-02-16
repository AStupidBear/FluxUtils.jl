using Flux.Optimise: Param, call
export syncparam!

function syncgrad(p::Param)
    function ()
        recvbuf = zero(p.Δ)
        MPI.Allreduce!(p.Δ, recvbuf, MPI.SUM, MPI.COMM_WORLD)
        p.Δ .= recvbuf ./ MPI.Comm_size
    end
end

function syncparam!(m)
    v = net2vec(m)
    MPI.Bcast!(v, 0, MPI.COMM_WORLD)
    vec2net!(m, v)
end

function Flux.Optimise.optimiser(ps, fs...)
    fs = (syncgrad, fs...)
    ps = [Param(p) for p in ps]
    fs = map(ps) do p
        os = map(f -> f(p), fs)
        () -> foreach(call, os)
    end
    () -> foreach(call, fs)
end
