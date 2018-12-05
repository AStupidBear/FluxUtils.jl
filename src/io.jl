export savenet, loadnet!

function loadnet!(m, src)
    BSON.@load src weights
    Flux.loadparams!(m, weights)
    return nothing
end

function savenet(dst, m)
    weights = Flux.data.(params(cpu(m)))
    BSON.@save dst weights
    return nothing
end
