using Base: Generator, product
export FluxNet

abstract type FluxNet end

part(x) = x

@require MPI begin
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
end

function rebatch(x, batchsize)
    nb, nt = size(x, 2), size(x, 3)
    n = batchsize ÷ nb
    n > 1 || return x
    nt′, nb′ = nt ÷ n, nb * n
    xt = view(x, :, :, OneTo(nt′ * n))
    xp = PermutedDimsArray(xt, [1, 3, 2])
    xr = reshape(xp, :, nt′, nb′)
    PermutedDimsArray(xr, [1, 3, 2])
end

function xy2data(x, y, batchsize, seqsize)
    x, y, batchsize, seqsize = rand(10, 100, 1000), rand(1, 100, 1000), 100, 10
    x = rebatch(part(x), batchsize)
    y = rebatch(part(y), batchsize)
    titr = indbatch(indices(x, 3), seqsize)
    bitr = indbatch(indices(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        ts, bs = args
        xs = [gpu(x[:, bs, t]) for t in ts]
        ys = [gpu(y[:, bs, t]) for t in ts]
        return xs, ys
    end
end

function fit!(m::FluxNet, x, y; cb = [])
    data = xy2data(x, y, m.batchsize, m.seqsize)
    Flux.@epochs m.epochs Flux.train!(m, m.loss, data, m.opt; cb = [cugc, cb...])
end

function predict!(ŷ, m::FluxNet, x)
    x = rebatch(x, m.batchsize)
    ŷ = rebatch(ŷ, m.batchsize)  
    mf = forwardmode(m)
    for bs in indbatch(indices(x, 2), m.batchsize)
        for t in 1:size(x, 3)
            ŷ[:, bs, t] = cpu(mf(gpu(x[:, bs, t])))
        end
    end
end
