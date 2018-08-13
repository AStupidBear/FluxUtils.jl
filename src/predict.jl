function predseq(m, xs, catseqdim; reset::Bool = true)
    ys = m.(xs) # for y in ys, y: 1xB
    zs = [veccat(catseqdim, [Flux.data(cpu(ys[t][n])) for t in 1:length(ys)]) for n in 1:length(first(ys))]
    reset && Flux.reset!(m)
    return zs
end

function predseqbatch(m, Xs, catseqdim = 1; reset::Bool = true)
    m′ = forwardmode(m)
    Ys = [predseq(m′, xs, catseqdim; reset = reset) for xs in Xs]
    ys = first(Ys)
    catdims = ndims.(ys)
    Zs = [cat(catdims[n], [Ys[b][n] for b in 1:length(Ys)]...) for n in 1:length(ys)]
    return Zs
end