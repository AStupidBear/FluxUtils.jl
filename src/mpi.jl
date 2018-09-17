using Flux.Optimise: Param, call

myrank() = MPI.Comm_rank(MPI.COMM_WORLD)

worldsize() = MPI.Comm_size(MPI.COMM_WORLD)

function syncgrad(p::Param)
    function ()
        recvbuf = zeros(p.Δ)
        MPI.Allreduce!(p.Δ, recvbuf, MPI.SUM, MPI.COMM_WORLD)
        p.Δ .= recvbuf ./ worldsize()
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

function part(x)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    if size(x, 3) > size(x, 2)
        c = Flux.chunk(indices(x, 3), size)
        view(x, :, :, c)
    else
        c = Flux.chunk(indices(x, 2), size)
        view(x, :, c, :)
    end
end