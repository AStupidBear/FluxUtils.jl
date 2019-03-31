using Flux.Tracker: istracked

export notrack

Flux.mapleaves(f, x::AbstractArray{<:Number}) = f(x)

function notrack(m)
    keep = Ref(true)
    Flux.prefor(p -> istracked(p) && (keep[] = false), m)
    keep[] ? m : mapleaves(Flux.data, m)
end
