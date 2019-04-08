export notrack

Flux.mapleaves(f, x::AbstractArray{<:Number}) = f(x)

function notrack(m)
    keep = Ref(true)
    prefor(p -> istracked(p) && (keep[] = false), m)
    keep[] ? m : mapleaves(data, m)
end
