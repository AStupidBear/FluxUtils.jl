using Flux: CuArrays

using Flux.CuArrays: CuArray, CUBLAS

# TODO: move this to Flux.jl
function Base.:(*)(A::CuArrays.CuMatrix, B::OneHotMatrix{CuArrays.CuArray{OneHotVector,1}})
    I = CuArrays.CuArray{UInt32, 1}(B.data.buf, 2 .* B.data.dims, offset = B.data.offset)[1:2:end]
    A[:, Array(I)]
end
