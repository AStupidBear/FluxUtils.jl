using Flux.Tracker: TrackedArray, track

export cugc, vecnorm2

cugc() = GC.gc(true)

for f in [:vcat, :hcat]
    @eval begin
        Base.$f(a::TrackedArray, b::SubArray) = track($f, a, b)
        Base.$f(a::SubArray, b::TrackedArray) = track($f, a, b)
    end
end

vecnorm2(x::TrackedArray, p::Real = 2) = sum(abs2.(x))

Flux.gpu(x) = Flux.mapleaves(identity, x)

function Flux.Tracker.ngradient(f, xs::AbstractArray...; δ = sqrt(eps()))
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        tmp = x[i]
        x[i] = tmp - δ / 2
        y1 = f(xs...)
        x[i] = tmp + δ / 2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2 - y1) / δ
    end
    return grads
end