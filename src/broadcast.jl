export +ᵇ, -ᵇ, *ᵇ, /ᵇ, ^ᵇ

+ᵇ(xs...) = broadcast(+, xs...)
-ᵇ(x, y) = x .- y
*ᵇ(x, y) = x .* y
/ᵇ(xs...) = x ./ y
^ᵇ(x, y) = x.^y

using Base.Broadcast: Broadcasted, materialize, broadcasted
using Flux.Tracker: data, tracker, unbroadcast, track, Call, TrackedStyle, broadcast_rebuild 
import Flux.Tracker: ∇broadcast

@inline function ∇broadcast(f::typeof(+), args::Vararg{Any, N}) where {N}
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified ∇broadcast for +"
        Δargs = ntuple(i -> Δ, Val(N))
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end

@inline function ∇broadcast(f::typeof(-), args::Vararg{Any, 2})
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified ∇broadcast for -"
        Δargs = (Δ, -Δ)
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end

@inline function ∇broadcast(f::typeof(*), args::Vararg{Any, 2})
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        @debug "Modified ∇broadcast for *"
        x1, x2 = args
        Δargs = (Δ .* x2, Δ .* x1)
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    track(Call(back, tracker.(args)), y)
end

function bc2ex(bc::Broadcasted, n = Ref(1))
    Expr(:call, Symbol(bc.f), [bc2ex(arg, n) for arg in bc.args]...)
end

bc2ex(arg, n) = (ex = Symbol("x$(n[])"); n[] += 1; ex)

function ∇bc(bc, n = Ref(1))
    ex = bc2ex(bc, n)
    args = [Symbol("x$i") for i in 1:(n[] - 1)]
    @static if VERSION < v"1.0"
        ∇ex = Calculus.differentiate(ex, args)
        @debug "using Calculus.jl to differentiate broadcasting" ∇ex
        ∇f = eval(Expr(:->, Expr(:tuple, args...), Expr(:tuple, ∇ex...)))
    else
        @debug "Using Zygote.jl to differentiate broadcasting" ∇ex
        f = eval(Expr(:->, Expr(:tuple, args...), ex))
        ∇f = (xs...) -> Zygote.gradient(f, xs...)
        global haha = ∇f
    end
    return ∇f
end

const gradable_symbols = [:+; :-; :*; :/; :^; first.(Calculus.symbolic_derivatives_1arg())]

function gradable(bc::Broadcasted)
    Symbol(bc.f) in gradable_symbols && all(gradable, bc.args)
end

gradable(arg) = true

function Base.Broadcast.materialize(bc::Broadcasted{TrackedStyle})
    bc1 = Broadcast.flatten(bc)
    bc2 = Broadcast.flatten(broadcast_rebuild(bc))
    if gradable(bc)
        ∇broadcast((bc2.f, ∇bc(bc)), bc1.args...)
    else
        ∇broadcast(bc2.f, bc1.args...)
    end
end

@inline function ∇broadcast(f∇f::Tuple{F, G}, args::Vararg{Any, N}) where {F, G, N}
    f, ∇f = f∇f
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        Δy = ∇f.(args...)
        Δargs = ntuple(i -> getindex.(Δy, i), Val(N))
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    # So we can return non-tracked arrays
    track(Call(back, tracker.(args)), y)
end