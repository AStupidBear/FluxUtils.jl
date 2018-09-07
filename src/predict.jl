function predseq(m, xs, catseqdim; reset::Bool = true)
    ys = m.(xs) # for y in ys, y: 1xB
    zs = [cat([Flux.data(cpu(ys[t][n])) for t in 1:length(ys)], catseqdim) for n in 1:length(first(ys))]
    reset && Flux.reset!(m)
    return zs
end

function predseqbatch(m, Xs, catseqdim = 1; reset::Bool = true)
    !reset && length(Xs) > 1 && error("batchsize is too small")
    m′ = reset ? forwardmode(m) : m
    Ys = [predseq(m′, xs, catseqdim; reset = reset) for xs in Xs]
    ys = first(Ys)
    catdims = ndims.(ys)
    Zs = [cat(catdims[n], [Ys[b][n] for b in 1:length(Ys)]...) for n in 1:length(ys)]
    return Zs
end