indbatch(x, b) = (C = ccount(x); i:min(i + b -1, C) for i in 1:b:C)

minibatch(x, batchsize) = Any[cview(x, ind) for ind in indbatch(x, batchsize)]

minibatch(x, y, batchsize) = Any[(cview(x, ind), cview(y, ind)) for ind in indbatch(x, batchsize)]

function tupseqbatch(Xs, batchsize, by = identity)
    Xs′ = (by.(xs) for xs in minibatch(Xs, batchsize))
    Xs′′ = ([cview(xs, i) for i in 1:ccount(xs)] for xs in Xs′)
end