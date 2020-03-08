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

using Flux.Optimise: StopException

macro interrupts(ex)
    :(try $(esc(ex))
        catch e
        e isa StopException || rethrow()
        throw(e)
        end)
end

function Flux.Optimise.train!(m, loss, data, opt, seqend; runback = true, 
                        runopt = true, cb = [], desc = "", kws...)
    cb = runall([cugc, cb...])
    l, nb = 0f0, 0
    ∇l = zero(net2vec(m))
    prog = Progress(length(data) + 1, desc = desc)
    for (n, (dn, se)) in enumerate(zip(data, seqend))
        ln = loss(m, dn...)
        runopt && next!(prog, showvalues = [(:loss, trunc4(ln.data))])
        isinf(ln) && error("Loss is Inf")
        isnan(ln) && error("Loss is NaN")
        runback && @interrupts back!(ln)
        l += Flux.data(ln)
        runback && (∇l .+= net2grad(m))
        nb += 1
        runopt && for x in params(m)
            update!(opt, x, Tracker.grad(x))
            x.tracker.grad = Tracker.zero_grad!(x.tracker.grad)
        end
        any(se[end]) ? reset!(m) : truncate!(m)
        cb()
    end
    l, ∇l =  l / nb, ∇l ./ nb
    runopt && next!(prog, showvalues = [(:avgloss, trunc4(l))])
    return l, ∇l
end
