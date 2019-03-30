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