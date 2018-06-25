function loadnet!(m, file)
    BSON.@load file weights
    Flux.loadparams!(m, weights)
    return nothing
end

function savenet(m, file)
    weights = Flux.data.(params(cpu(m)))
    BSON.@save file weights
    return nothing
end