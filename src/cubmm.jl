using CuArrays: CuArray, CUBLAS

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