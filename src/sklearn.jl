using Base: Generator, product

abstract type FluxNet end

part(x) = x

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

function datagen(x, batchsize, seqsize)
    x = rebatch(part(x), batchsize)
    titr = indbatch(indices(x, 3), seqsize)
    bitr = indbatch(indices(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        ts, bs = args
        xs = [gpu(x[:, bs, t]) for t in ts]
        return xs
    end
end

function datagen(x, batchsize)
    x = rebatch(part(x), batchsize)
    titr = indices(x, 3)
    bitr = indbatch(indices(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        t, bs = args
        view(x, :, bs, t)
    end
end

datagen(x::Tuple, args...) = zip(datagen.(x, args...)...)

function fit!(m::FluxNet, x, y; sample_weight = nothing, cb = [])
    dx = datagen(x, m.batchsize, m.seqsize)
    dy = datagen(y, m.batchsize, m.seqsize)
    if sample_weight == nothing
        data = zip(dx, dy)
    else
        dw = datagen(y, m.batchsize, m.seqsize)
        data = zip(dx, dy, dw)
    end
    Flux.@epochs m.epochs Flux.train!(m, m.loss, data, m.opt; cb = [cugc, cb...])
end

function predict!(ŷ, m::FluxNet, x)
    fill!(ŷ, 0f0)
    data = zip(datagen(x, m.batchsize), datagen(ŷ, m.batchsize))
    mf = forwardmode(m)
    for (xi, yi) in data
        copy!(yi, cpu(mf(gpu(xi))))
    end
    return ŷ
end

Base.fill!(As::Tuple, x) = fill!.(As, x)

Base.copy!(dests::Tuple, srcs::Tuple) = copy!.(dests, srcs)