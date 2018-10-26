using Base: Generator, product
using Flux: chunk
using Compat: argmax

export FluxNet, xy2data, datagen, part

abstract type FluxNet end

myrank() = max(0, myid() - 2)

worldsize() = nworkers()

function part(x)
    d = argmax(size(x))
    is = chunk(axes(x, d), worldsize())
    i = UnitRange(extrema(is[myrank() + 1])...)
    inds = ntuple(x -> x == d ? i : (:), ndims(x))
    view(x, inds...)
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

function datagen(x, batchsize, seqsize; parl = true)
    x = parl ? part(x) : x
    x = rebatch(x, batchsize)
    titr = indbatch(axes(x, 3), seqsize)
    bitr = indbatch(axes(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        ts, bs = args
        xs = [gpu32(x[:, bs, t]) for t in ts]
        return xs
    end
end

function datagen(x, batchsize; parl = true)
    x = parl ? part(x) : x
    x = rebatch(x, batchsize)
    titr = axes(x, 3)
    bitr = indbatch(axes(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        t, bs = args
        view(x, :, bs, t)
    end
end

datagen(x::Tuple, args...; kwargs...) = zip(datagen.(x, args...; kwargs...)...)

function fit!(m::FluxNet, x, y; sample_weight = nothing, parl = true, cb = [])
    checkdims(x, y)
    dx = datagen(x, m.batchsize, m.seqsize, parl = parl)
    dy = datagen(y, m.batchsize, m.seqsize, parl = parl)
    if sample_weight == nothing
        data = zip(dx, dy)
    else
        checkdims(sample_weight)
        w = sample_weight
        rmul!(w, length(w) / sum(w))
        dw = datagen(w, m.batchsize, m.seqsize)
        data = zip(dx, dy, dw)
    end
    Flux.@epochs m.epochs Flux.train!(m, m.loss, data, m.opt; cb = [cugc, cb...])
end

function predict!(ŷ, m::FluxNet, x; reset = true)
    checkdims(x, ŷ)
    fill!(ŷ, 0f0)
    dx = datagen(x, m.batchsize, parl = false)
    dy = datagen(ŷ, m.batchsize, parl = false)
    mf = reset ? forwardmode(m) : m
    for (xi, yi) in zip(dx, dy)
        copyto!(yi, forwardmode(cpu(mf(gpu32(xi)))))
    end
    return ŷ
end

Base.fill!(As::Tuple, x) = fill!.(As, x)

Compat.copyto!(dests::Tuple, srcs::Tuple) = copyto!.(dests, srcs)

checkdims(xs...) = Flux.prefor(x -> x isa AbstractArray && ndims(x) != 3 && error("ndims should be 3"), xs)