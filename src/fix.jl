@require CuArrays function Base.:(*)(A::CuArrays.CuMatrix, B::Flux.OneHotMatrix{CuArrays.CuArray{Flux.OneHotVector,1}})
    I = CuArrays.CuArray{UInt32, 1}(B.data.buf, B.data.offset, 2 .* B.data.dims)[1:2:end]
    A[:, Array(I)]
end

@require CuArrays cugc() = (gc(); CuArrays.reclaim(true))

