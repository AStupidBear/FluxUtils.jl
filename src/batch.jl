indbatch(x, b) = (C = ccount(x); i:min(i + b -1, C) for i in 1:b:C)

minibatch(x, batchsize) = Any[cview(x, ind) for ind in indbatch(x, batchsize)]

minibatch(x, y, batchsize) = Any[(cview(x, ind), cview(y, ind)) for ind in indbatch(x, batchsize)]

indbatchseq(x, b) = (T = ccount(x) ÷ b; t:T:T*b for t in 1:T)

minibatchseq(x, batchsize) =  Any[cview(x, ind) for ind in indbatchseq(x, batchsize)]

minibatchseq(x, y, batchsize) =  Any[(cview(x, ind), cview(y, ind)) for ind in indbatchseq(x, batchsize)]
