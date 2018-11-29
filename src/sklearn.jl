using Base: Generator, product
using Flux: chunk
using Compat: argmax

export FluxNet, xy2data, datagen, part

abstract type FluxNet end

function part(x, n = myid() - 1, N = nworkers(); dim = ndims(x))
    (n < 1 || size(x)[dim] < N) && return x
    is = chunk(axes(x, dim), N)
    i = UnitRange(extrema(is[n])...)
    inds = ntuple(x -> x == dim ? i : (:), ndims(x))
    view(x, inds...)
end

mpipart(x) = part(x, myid(), nprocs())

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

function datagen(x, batchsize, seqsize; partf = part)
    x = rebatch(partf(x), batchsize)
    titr = indbatch(axes(x, 3), seqsize)
    bitr = indbatch(axes(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        ts, bs = args
        xs = [gpu32(x[:, bs, t]) for t in ts]
        return xs
    end
end

function datagen(x, batchsize; partf = part)
    x = rebatch(partf(x), batchsize)
    titr = axes(x, 3)
    bitr = indbatch(axes(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        t, bs = args
        view(x, :, bs, t)
    end
end

datagen(x::Tuple, args...; kwargs...) = zip(datagen.(x, args...; kwargs...)...)

function fit!(m::FluxNet, x, y; sample_weight = nothing, cb = [])
    checkdims(x, y)
    dx = datagen(x, m.batchsize, m.seqsize, partf = mpipart)
    dy = datagen(y, m.batchsize, m.seqsize, partf = mpipart)
    if sample_weight == nothing
        data = zip(dx, dy)
    else
        checkdims(sample_weight)
        w = sample_weight
        rmul!(w, length(w) / sum(w))
        dw = datagen(w, m.batchsize, m.seqsize, partf = mpipart)
        data = zip(dx, dy, dw)
    end
    Flux.@epochs m.epochs Flux.train!(m, m.loss, data, m.opt; cb = [cugc, cb...])
end

function predict!(ŷ, m::FluxNet, x; reset = true)
    checkdims(x, ŷ)
    fill!(ŷ, 0f0)
    dx = datagen(x, m.batchsize, partf = identity)
    dy = datagen(ŷ, m.batchsize, partf = identity)
    mf = reset ? forwardmode(m) : m
    for (xi, yi) in zip(dx, dy)
        copyto!(yi, forwardmode(cpu(mf(gpu32(xi)))))
    end
    return ŷ
end

Base.fill!(As::Tuple, x) = fill!.(As, x)

Compat.copyto!(dests::Tuple, srcs::Tuple) = copyto!.(dests, srcs)

checkdims(xs...) = Flux.prefor(x -> x isa AbstractArray && ndims(x) != 3 && error("ndims should be 3"), xs)
