export plog, @pepochs

function plog(name, val, color = :blue)
    str = @sprintf("worker: %d, %s: %.4f\n", myid(), name, val)
    printstyled(str, color = color)
    flush(stdout)
end

macro pepochs(n, ex)
  :(for i = 1:$(esc(n))
      plog("epoch", i, :green)
      $(esc(ex))
      cugc()
    end)
end

function Flux.Optimise.train!(m, loss, data, opt; logintvl = 10, cb = [])
    cb = runall(cb)
    opt = runall(opt)
    ltot, nbatch = 0f0, 0
    logcb = throttle(plog, logintvl)
    for (i, d) in enumerate(data)
        l = loss(m, d...)
        if i % size(data)[1] == 0
            Flux.reset!(m)
        else
            Flux.truncate!(m)
        end
        ltot += Flux.data(l)
        nbatch += 1
        logcb("loss", l)
        isinf(l) && error("Loss is Inf")
        isnan(l) && error("Loss is NaN")
        @interrupts back!(l)
        opt()
        cb() == :stop && break
    end
    plog("avgloss", ltot / nbatch, :yellow)
end
