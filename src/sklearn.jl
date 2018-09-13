export FluxNet
abstract type FluxNet end

function xy2data(x, y, batchsize, seqsize = 1)
    if ndims(x) == 3
        data = ((gpu.(eachcol(x[:, ib, is])), gpu.(eachcol(y[:, ib, is])))
                for is in indbatch(1:size(x, 3), seqsize)
                for ib in indbatch(1:size(x, 2), batchsize))
    else
        data = ((gpu.(eachcol(x[:, ib])), gpu.(eachcol(y[:, ib])))
                for ib in indbatch(1:size(x, 2), batchsize))
    end
end

function fit!(m::FluxNet, x, y; cb = [])
    data = xy2data(x, y, m.batchsize, m.seqsize)
    Flux.@epochs m.epochs Flux.train!(m, m.loss, data, m.opt; cb = [cugc, cb...])
end

function predict!(ŷ, m::FluxNet, x)
    mf = forwardmode(m)
    for ib in indbatch(1:size(x, 2), mf.batchsize)
        if ndims(x) == 3
            for t in 1:size(x, 3)
                ŷ[:, ib, t] = cpu(mf(gpu(x[:, ib, t])))
            end
        else
            ŷ[:, ib] = cpu(mf(gpu(x[:, ib])))
        end
    end
end
