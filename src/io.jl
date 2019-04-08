export savenet, loadnet!

function loadnet!(m, src)
    BSON.@load src weights
    loadparams!(m, weights)
    return nothing
end

function savenet(dst, m)
    weights = data.(params(cpu(m)))
    BSON.@save dst weights
    return nothing
end
