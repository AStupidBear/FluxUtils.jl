function plog(name, val, color = :blue)
    str = @sprintf("\nworker: %d, %s: %.4f\n", myid(), name, val)
    printstyled(str, color = color)
    flush(stdout)
end

function Flux.Optimise.train!(m, loss, data, opt; logintvl = 10,
                runback = true, runopt = true, cb = [], kws...)
    cb = runall([cugc, cb...])
    opt = runall(opt)
    l, nb = 0f0, 0
    ∇l = zero(net2vec(m))
    logcb = throttle(plog, logintvl)
    @showprogress for (n, dn) in enumerate(data)
        ln = loss(m, dn...)
        logcb("loss", ln)
        isinf(ln) && error("Loss is Inf")
        isnan(ln) && error("Loss is NaN")
        runback && @interrupts back!(ln)
        l += Flux.data(ln)
        runback && (∇l .+= net2grad(m))
        nb += 1
        runopt && opt()
        Flux.truncate!(m)
        cb()
    end
    l /= nb
    rmul!(∇l, 1 / nb)
    plog("avgloss", l, :yellow)
    return l, ∇l
end
