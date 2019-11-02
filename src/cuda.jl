using Flux: CuArrays

using Flux.CuArrays: CuArray, CUBLAS

# TODO: move this to Flux.jl
function Base.:(*)(A::CuArrays.CuMatrix, B::OneHotMatrix{CuArrays.CuArray{OneHotVector,1}})
    I = CuArrays.CuArray{UInt32, 1}(B.data.buf, 2 .* B.data.dims, offset = B.data.offset)[1:2:end]
    A[:, Array(I)]
end

# TODO: remove this once CuArrays is updated
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = CuArrays.CUDAnative.pow(x, Int32(p))

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))

    @eval begin
        function batched_gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3}, beta::($elty), C::CuArray{$elty, 3})
            CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
        end

        function batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            CUBLAS.gemm_strided_batched(transA, transB, alpha, A, B)
        end

        function batched_gemm(transA::AbstractChar, transB::AbstractChar, A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            CUBLAS.gemm_strided_batched(transA, transB, A, B)
        end
    end
end