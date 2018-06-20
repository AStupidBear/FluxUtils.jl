function predict_seq(m, xs, catseqdim)
    ys = m.(xs) # for y in ys, y: 1xB
    zs = [cat(catseqdim, [Flux.data(cpu(ys[t][n])) 
            for t in 1:length(ys)]...) 
            for n in 1:length(first(ys))]
    Flux.reset!(m)
    return zs
end

function predict_batch(m, Xs, batchsize, by = identity, catseqdim = 1)
    m′ = forwardmode(m)
    Xs′ = batchtupleseq(Xs, batchsize, by)
    Ys = [predict_seq(m′, xs, catseqdim) for xs in Xs′]
    ys = first(Ys)
    catdims = ndims.(ys)
    Zs = [cat(catdims[n], [Ys[b][n] for b in 1:length(Ys)]...) for n in 1:length(ys)]
    Flux.reset!(m)
    return Zs
end