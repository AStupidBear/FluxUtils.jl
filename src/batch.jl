indbatch(x, b, offset = 0) = (C = ccount(x); min(i + offset, C):min(i + offset + b -1, C) for i in 1:b:C)

minibatch(x, batchsize) = Any[cview(x, ind) for ind in indbatch(x, batchsize)]

minibatch(x, y, batchsize) = Any[(cview(x, ind), cview(y, ind)) for ind in indbatch(x, batchsize)]

function tupseqbatch(Xs, batchsize, by = identity)
    Xs′ = (by.(xs) for xs in minibatch(Xs, batchsize))
    Xs′′ = ([gpu(cview(xs, i)) for i in 1:ccount(xs)] for xs in Xs′)
end