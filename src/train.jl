export plog, @pepochs

function plog(name, val, color = :blue)
    str = @sprintf("Rank: %d, %s: %.4f\n", myrank(), name, val)
    print_with_color(color, str)
    flush(STDOUT)
end

macro pepochs(n, ex)
  :(for i = 1:$(esc(n))
      info("Rank: $(myrank()), Epoch $i")
      flush(STDOUT)
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
        logcb("Loss", l)
        isinf(l) && error("Loss is Inf")
        isnan(l) && error("Loss is NaN")
        @interrupts back!(l)
        opt()
        cb() == :stop && break
    end
    plog("AvgLoss", ltot / nbatch, :yellow)
end