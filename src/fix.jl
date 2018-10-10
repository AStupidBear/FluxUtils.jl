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