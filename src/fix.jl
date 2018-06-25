@require CuArrays function Base.:(*)(A::CuArrays.CuMatrix, B::Flux.OneHotMatrix{CuArrays.CuArray{Flux.OneHotVector,1}})
    I = CuArrays.CuArray{UInt32, 1}(B.data.buf, B.data.offset, 2 .* B.data.dims)[1:2:end]
    A[:, Array(I)]
end

export cugc
cugc() = gc()
@require CuArrays @suppress cugc() = (gc(); CuArrays.reclaim(true))

using Flux.Tracker: TrackedArray, track

for f in [:vcat, :hcat]
  @eval begin
    Base.$f(a::TrackedArray, b::SubArray) = track($f, a, b)
    Base.$f(a::SubArray, b::TrackedArray) = track($f, a, b)
    @require CuArrays begin
        Base.$f(a::TrackedArray, b::CuArrays.CuArray) = track($f, a, b)
        Base.$f(a::CuArrays.CuArray, b::TrackedArray) = track($f, a, b)
    end
  end
end

export vecnorm2
vecnorm2(x::TrackedArray, p::Real = 2) = sqrt(sum(abs.(x).^2 .+ eps(0f0)))