export notrack, cugc

function cugc()
    GC.gc()
    isdefined(Flux, :CuArrays) &&
    Flux.CuArrays.BinnedPool.reclaim()
end

function notrack(m)
    keep = Ref(true)
    prefor(p -> istracked(p) && (keep[] = false), m)
    keep[] ? m : mapleaves(data, m)
end

# TODO: remove this once the zygote branch is released
if !isdefined(Flux, :Zygote)
    function Flux.Optimise.apply!(o::WeightDecay, x, Δ)
        wd = o.wd
        Δ .+= wd .* data(x)
    end
end