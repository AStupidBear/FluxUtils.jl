function cleargrad!(m)
    for p in Flux.params(m)
        p.grad .= 0
    end
end

function gradseq(m, loss, xs)
    l = loss(xs)
    Flux.back!(l)
    Δ = net2grad(m)
    cleargrad!(m)
    Flux.reset!(m)
    return Δ
end

function gradseqbatch(m, loss, Xs)
    Δs = [ccount(first(xs)) .* gradseq(m, loss, xs) for xs in Xs]
    Δ = sum(Δs) ./ ccount(Xs)
    Flux.reset!(m)
    return Δ
end

function lossseqbatch(loss, Xs)
    l = sum(Flux.data(loss(xs)) * length(xs) * ccount(first(xs)) for xs in Xs)
    l /= sum(length(xs) * ccount(first(xs)) for xs in Xs)
end