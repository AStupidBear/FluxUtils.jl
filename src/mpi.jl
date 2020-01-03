using .MPI: SUM, Comm_size, COMM_WORLD, Allreduce!, Bcast!

export syncparam!

function Flux.Optimise.update!(opt, x, x̄)
    Δ = data(x̄)
    if MPI.Initialized()
        Δ′ = zero(Δ)
        Allreduce!(Δ, Δ′, SUM, COMM_WORLD)
        Δ .= Δ′ ./ Comm_size(COMM_WORLD)
    end
    update!(x, -apply!(opt, x, Δ))
end

function syncparam!(m)
    if MPI.Initialized()
        v = net2vec(m)
        Bcast!(v, 0, COMM_WORLD)
        vec2net!(m, v)
    end
end