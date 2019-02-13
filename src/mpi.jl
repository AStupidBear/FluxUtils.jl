using Flux.Optimise: Param, call

function syncgrad(p::Param)
    function ()
        recvbuf = zero(p.Δ)
        MPI.Allreduce!(p.Δ, recvbuf, MPI.SUM, MPI.COMM_WORLD)
        p.Δ .= recvbuf ./ nworkers()
    end
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
