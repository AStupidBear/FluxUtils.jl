using Base: Generator, product

export rebatch, datagen, Estimator, seqloss

function part(x::AbstractArray, n, N; dims = 0)
    nd = ndims(x)
    dims = dims > 0 ? dims :
          dims < 0 ? nd + 1 - dims :
          size(x, nd - 1) > 5 * size(x, nd) ?
          nd - 1 : nd
    size(x)[dims] < N && return x
    is = chunk(1:size(x, dims), N)
    i = UnitRange(extrema(is[n])...)
    inds = ntuple(x -> x == dims ? i : (:), ndims(x))
    view(x, inds...)
end

function part(x; dims = 0)
    if @isdefined(MPI) && MPI.Initialized()
        part(x, myrank() + 1, nhosts())
    else
        return x
    end
end

fieldvalues(x) = [getfield(x, s) for s in fieldnames(typeof(x))]

part(obj, T) = T([isa(x, AbstractArray) ? part(x) : x for x in fieldvalues(x)])

function rebatch(x::AbstractMatrix, batchsize)
    N, T = size(x, 1), size(x, 2)
    n = batchsize ÷ N
    (n <= 1 || T <= n) && return x
    T′, N′ = T ÷ n, N * n
    xv = view(x, :, 1:(T′ * n))
    xp = PermutedDimsArray(xv, [2, 1])
    xr = reshape(xp, T′, N′)
    PermutedDimsArray(xr, [2, 1])
end

function rebatch(x::AbstractArray{<:Any, 3}, batchsize)
    N, T = size(x, 2), size(x, 3)
    n = batchsize ÷ N
    (n <= 1 || T <= n) && return x
    T′, N′ = T ÷ n, N * n
    xv = view(x, :, :, 1:(T′ * n))
    xp = PermutedDimsArray(xv, [1, 3, 2])
    xr = reshape(xp, :, T′, N′)
    PermutedDimsArray(xr, [1, 3, 2])
end

function datagen(x, batchsize, seqsize; partf = part, trans = identity, cb = identity)
    x = cb(rebatch(partf(x), batchsize))
    titr = indbatch(1:size(x, 3), seqsize)
    bitr = indbatch(1:size(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        ts, bs = args
        [trans(view(x, :, bs, t)) for t in ts]
    end
end

function datagen(x, batchsize; partf = part, trans = identity)
    x = rebatch(partf(x), batchsize)
    titr = 1:size(x, 3)
    bitr = indbatch(1:size(x, 2), batchsize)
    Generator(product(titr, bitr)) do args
        t, bs = args
        trans(view(x, :, bs, t))
    end
end

datagen(x::Tuple, args...; kwargs...) = zip(datagen.(x, args...; kwargs...)...)

Base.fill!(As::Tuple, x) = fill!.(As, x)

Base.copyto!(dests::Tuple, srcs::Tuple) = copyto!.(dests, srcs)

checkdims(xs...) = prefor(x -> x isa AbstractArray && ndims(x) != 3 && error("ndims should be 3"), xs)

mutable struct Estimator{M, L, O, C}
    model::M
    loss::L
    opt::O
    spec::C
end

function Base.show(io::IO, est::Estimator)
    io = IOContext(io, :compact => true)
    println(io, "model:")
    for s in fieldnames(typeof(est.model))
        x = getfield(est.model, s)
        x = x == nothing ? "nithing" : x
        println(io, ' '^2, s, ": ", x)
    end
    println(io, "loss: ", repr("text/plain", est.loss))
    println(io, "opt: ", repr("text/plain", est.opt))
    println(io, "spec: ", est.spec)
end

@treelike Estimator

adaptor(m) = x -> Flux.adapt(typeof(data(first(params(m)))), x)

function fit!(est::Estimator, x, y, w = nothing; kws...)
    @unpack model, loss, opt, spec = est
    @unpack epochs, batchsize, seqsize = spec
    haskey(kws, :epochs) && @unpack epochs = kws
    runopt = haskey(kws, :runopt) ? kws[:runopt] : true
    runopt && @isdefined(MPI) && syncparam!(est)
    dx = datagen(x, batchsize, seqsize, partf = part, trans = adaptor(est) ∘ copy)
    dy = datagen(y, batchsize, seqsize, partf = part, trans = adaptor(est) ∘ copy)
    seqend = datagen(fill(false, 1, size(x)[2:end]...), batchsize, seqsize, 
                    cb = z -> (z[:, :, end] .= true; z))
    if w == nothing
        data = zip(dx, dy)
    else
        rmul!(w, 1 / mean(w))
        dw = datagen(w, batchsize, seqsize, partf = part)
        data = zip(dx, dy, dw)
    end
    local l, ∇l
    for n in 1:epochs
        desc = !@isdefined(MPI) ? @sprintf("epoch-%d ", n) :
                @sprintf("rank-%d,epoch-%d ", myrank(), n)
        l, ∇l = train!(model, loss, data, opt, seqend; desc = desc, kws...)
    end
    return l, ∇l
end

function predict!(ŷ, est::Estimator, x; reset = true)
    @unpack model, spec = est
    @unpack batchsize, seqsize = spec
    model = notrack(model)
    fill!(ŷ, 0f0) # in case of partial copy
    dx = datagen(x, batchsize, partf = identity, trans = adaptor(est) ∘ copy)
    dy = datagen(ŷ, batchsize, partf = identity)
    reset && reset!(model)
    for (xi, yi) in zip(dx, dy)
        copyto!(yi, notrack(cpu(model(xi))))
    end
    return ŷ
end

function seqloss(loss)
    function (m, xs, ys)
        l, T = 0f0, length(xs)
        for t in 1:T
            x, y = xs[t], ys[t]
            l += loss(m(x), y)
        end
        return l / Float32(T)
    end
end
