import Flux.Optimise: update!, apply!

export syncparam!

function Flux.Optimise.update!(opt, x, x̄)
    Δ = Flux.data(x̄)
    Δ′ = zero(Δ)
    MPI.Allreduce!(Δ, Δ′, MPI.SUM, MPI.COMM_WORLD)
    Δ .= Δ′ ./ MPI.Comm_size(MPI.COMM_WORLD)
    update!(x, -apply!(opt, x, Δ))
end


function syncparam!(m)
    v = net2vec(m)
    MPI.Bcast!(v, 0, MPI.COMM_WORLD)
    vec2net!(m, v)
end

if VERSION < v"1.0"
    using Flux.Optimise: Param, call

    function syncgrad(p::Param)
        function ()
            recvbuf = zero(p.Δ)
            MPI.Allreduce!(p.Δ, recvbuf, MPI.SUM, MPI.COMM_WORLD)
            p.Δ .= recvbuf ./ MPI.Comm_size(MPI.COMM_WORLD)
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