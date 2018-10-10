export +ᵇ, -ᵇ, *ᵇ, /ᵇ, ^ᵇ

+ᵇ(xs...) = broadcast(+, xs...)
-ᵇ(x, y) = x .- y
*ᵇ(x, y) = x .* y
/ᵇ(xs...) = x ./ y
^ᵇ(x, y) = x.^y

using Flux.Tracker: data, tracker, unbroadcast, track, Call

@inline function Flux.Tracker.∇broadcast(f::typeof(+), args::Vararg{Any, N}) where {N}
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified Flux.Tracker.∇broadcast for +"
        Δargs = ntuple(i -> Δ, Val(N))
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end

@inline function Flux.Tracker.∇broadcast(f::typeof(-), args::Vararg{Any, 2})
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified Flux.Tracker.∇broadcast for -"
        Δargs = (Δ, -Δ)
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end

@inline function Flux.Tracker.∇broadcast(f::typeof(*), args::Vararg{Any, 2})
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified Flux.Tracker.∇broadcast for *"
        x1, x2 = args
        Δargs = (Δ .* x2, Δ .* x1)
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end