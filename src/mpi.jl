myrank() = MPI.Comm_rank(MPI.COMM_WORLD)

nhosts() = MPI.Comm_size(MPI.COMM_WORLD)

function Flux.Optimise.update!(opt, x, x̄)
    Δ = data(x̄)
    if MPI.Initialized()
        Δ′ = zero(Δ)
        MPI.Allreduce!(Δ, Δ′, MPI.SUM, MPI.COMM_WORLD)
        Δ .= Δ′ ./ nhosts()
    end
    update!(x, -apply!(opt, x, Δ))
end

function syncparam!(m)
    if MPI.Initialized()
        v = net2vec(m)
        MPI.Bcast!(v, 0, MPI.COMM_WORLD)
        vec2net!(m, v)
    end
end