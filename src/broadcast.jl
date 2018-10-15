export +ᵇ, -ᵇ, *ᵇ, /ᵇ, ^ᵇ

+ᵇ(xs...) = broadcast(+, xs...)
-ᵇ(x, y) = x .- y
*ᵇ(x, y) = x .* y
/ᵇ(xs...) = x ./ y
^ᵇ(x, y) = x.^y


using Base.Broadcast: Broadcasted, materialize, broadcasted
using Flux.Tracker: data, tracker, unbroadcast, track, Call, TrackedStyle, broadcast_rebuild, istracked
import Flux.Tracker: ∇broadcast
using MacroTools: postwalk

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

const ∇EX2∇f = Dict{Expr, Function}()
const ∇EX2COUNT = Dict{Expr, Int}()

function real2float(ex)
    postwalk(ex) do x
        isa(x, Real) ? Float32(x) : x
    end
end

function ∇bc(bc, n = Ref(1))
    ex = bc2ex(bc, n)
    args = [Symbol("x$i") for i in 1:(n[] - 1)]
    ∇exs = real2float.(Calculus.simplify.(Calculus.differentiate(ex, args)))
    @debug "using Calculus.jl to differentiate broadcasting" ∇exs
    ∇fs = Function[]
    for ∇ex in ∇exs
        ∇fex = Expr(:->, Expr(:tuple, args...), ∇ex)
        ∇f = haskey(∇EX2∇f, ∇fex) ? ∇EX2∇f[∇fex] : ∇EX2∇f[∇fex] = eval(∇fex)
        push!(∇fs, ∇f) 
    end
    return ∇fs
end

const gradable_symbols = [:+; :-; :*; :/; :^; first.(Calculus.symbolic_derivatives_1arg())]

function gradable(bc::Broadcasted)
    Symbol(bc.f) in gradable_symbols && all(gradable, bc.args)
end

gradable(arg) = true

function Base.Broadcast.materialize(bc::Broadcasted{TrackedStyle})
    bc1 = Broadcast.flatten(bc)
    bc2 = Broadcast.flatten(broadcast_rebuild(bc))
    if gradable(bc) && !in(bc2.f, [+, -, *])
        ∇broadcast((bc2.f, ∇bc(bc)), bc1.args...)
    else
        ∇broadcast(bc2.f, bc1.args...)
    end
end

@noinline function ∇broadcast(f∇f::Tuple{F, G}, args::Vararg{Any, N}) where {F, G, N}
    f, ∇fs = f∇f
    y = broadcast(f, data.(args)...)
    eltype(y) <: Real || return y
    eltype(y) == Bool && return y
    function back(Δ)
        Δargs = try
            Δargs = ntuple(i -> istracked(args[i]) ? Δ .* ∇fs[i].(args...) : zero(Δ), Val(N))
        catch
            @debug "invokelatest ∇fs[i]"
            ∇fs′ = ntuple(i -> (x...) -> Base.invokelatest(∇fs[i], x...), Val(N))
            Δargs = ntuple(i -> istracked(args[i]) ? Δ .* ∇fs′[i].(args...) : zero(Δ), Val(N))
        end
        dxs = map(unbroadcast, args, Δargs)
        return dxs
    end
    # So we can return non-tracked arrays
    track(Call(back, tracker.(args)), y)
end