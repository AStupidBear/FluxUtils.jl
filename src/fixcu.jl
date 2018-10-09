cugc() = (gc(true); CuArrays.reclaim(true))

function Base.:(*)(A::CuArrays.CuMatrix, B::Flux.OneHotMatrix{CuArrays.CuArray{Flux.OneHotVector,1}})
    I = CuArrays.CuArray{UInt32, 1}(B.data.buf, B.data.offset, 2 .* B.data.dims)[1:2:end]
    A[:, Array(I)]
end

for f in [:vcat, :hcat]
    @eval begin
        Base.$f(a::TrackedArray, b::CuArrays.CuArray) = track($f, a, b)
        Base.$f(a::CuArrays.CuArray, b::TrackedArray) = track($f, a, b)
    end
end

Flux.gpu(x) = Flux.mapleaves(CuArrays.cu, x)