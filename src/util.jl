export notrack, cugc, *ᶜ
export +ᵇ, -ᵇ, *ᵇ, /ᵇ, ^ᵇ

function cugc()
    GC.gc()
    isdefined(Flux, :CuArrays) &&
    Flux.CuArrays.reclaim(true)
end

function notrack(m)
    keep = Ref(true)
    prefor(p -> istracked(p) && (keep[] = false), m)
    keep[] ? m : mapleaves(data, m)
end

function *ᶜ(A, B) # tensor contraction
    Ar = reshape(A, :, size(A, ndims(A)))
    Br = reshape(B, size(B, 1), :)
    dims = (size(A)[1:end-1]..., size(B)[2:end]...)
    C = reshape(Ar * Br, dims...)
end

+ᵇ(xs...) = broadcast(+, xs...)
-ᵇ(x, y) = x .- y
*ᵇ(x, y) = x .* y
/ᵇ(xs...) = x ./ y
^ᵇ(x, y) = x.^y
