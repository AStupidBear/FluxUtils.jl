
export plog

function plog(str::AbstractString; kws...)
    str = @sprintf("worker: %d, %s\n", myid(), str)
    printstyled(str; kws...)
    flush(stdout)
end

trunc4(x) = isa(x, AbstractFloat) ? trunc(x, digits = 4) : x

function plog(nt::NamedTuple; kws...)
    strs = [@sprintf("%s: %s", k, trunc4(v)) for (k, v) in pairs(nt)]
    plog(join(strs, ", "); kws...)
end

function Flux.Optimise.train!(m, loss, data, opt; runback = true, 
                        runopt = true, cb = [], desc = "", kws...)
    cb = runall([cugc, cb...])
    opt = runall(opt)
    l, nb = 0f0, 0
    ∇l = zero(net2vec(m))
    prog = Progress(length(data) + 1, desc = desc)
    for (n, dn) in enumerate(data)
        ln = loss(m, dn...)
        next!(prog, showvalues = [(:loss, @sprintf("%.4f", ln))])
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
    l /= nb; rmul!(∇l, 1 / nb)
    next!(prog, showvalues = [(:avgloss, @sprintf("%.4f", l))])
    return l, ∇l
end
