export bmm, *ᶜ

function *ᶜ(A, B) # tensor contraction
    Ar = reshape(A, :, size(A, ndims(A)))
    Br = reshape(B, size(B, 1), :)
    dims = (size(A)[1:end-1]..., size(B)[2:end]...)
    C = reshape(Ar * Br, dims...)
end

bmm(A, B) = batched_gemm(A, B)

bmm(A::TrackedArray, B::AbstractArray) = track(bmm, A, B)
bmm(A::AbstractArray, B::TrackedArray) = track(bmm, A, B)
bmm(A::TrackedArray, B::TrackedArray) = track(bmm, A, B)

bmm(A, B) = batched_gemm('N', 'N', A, B)

@grad function bmm(A, B)
     bmm(data(A), data(B)),
     Δ -> (batched_gemm('N', 'T', Δ, data(B)),
         batched_gemm('T', 'N', data(A), Δ))
end

# TODO: use gemm_batch when mkl is available

function batched_gemm! end
function batched_gemm end

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
        function batched_gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3}, beta::($elty), C::AbstractArray{$elty, 3})
            @assert !BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            BLAS.chkstride1(A)
            BLAS.chkstride1(B)
            BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                     Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                     Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end

            C
        end
        function batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm!(transA, transB, alpha, A, B, zero($elty), similar(B, $elty, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(B, 3))))
        end
        function batched_gemm(transA::AbstractChar, transB::AbstractChar, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm(transA, transB, one($elty), A, B)
        end
    end
end