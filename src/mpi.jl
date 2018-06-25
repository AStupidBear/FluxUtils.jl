using Flux.Optimise: Param, call

@require MPI begin

myrank() = MPI.Comm_rank(MPI.COMM_WORLD)

worldsize() = MPI.Comm_size(MPI.COMM_WORLD)

function syncgrad(p::Param)
    function ()
        recvbuf = zeros(p.Δ)
        MPI.Allreduce!(p.Δ, recvbuf, MPI.SUM, MPI.COMM_WORLD)
        p.Δ .= recvbuf
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

end