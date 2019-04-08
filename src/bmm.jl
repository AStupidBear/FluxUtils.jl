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
