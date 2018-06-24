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

function gradbatch(m, loss, Xs)
    Δs = [ccount(xs[1]) .* gradseq(m, loss, xs) for xs in Xs]
    Δ = sum(Δs) ./ ccount(Xs)
    Flux.reset!(m)
    return Δ
end