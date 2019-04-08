export indbatch, minibatch

@generated function subslice(x::AbstractArray{T, N}) where {T, N}
    inds = ntuple(i -> (:), N - 1)
    :($inds)
end
subslice(x) = ntuple(i -> (:), ndims(x) - 1)

ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))
cview(a, i) = view(a, subslice(a)..., i)
cget(a, i) = getindex(a, subslice(a)..., i)
ccount(x::Tuple) = ccount(x[1])
for s in (:cget, :cview)
    @eval $s(as::Tuple, i) = tuple([$s(a, i) for a in as]...)
end

indbatch(x, b, offset = 0) = (C = ccount(x); min(i + offset, C):min(i + offset + b -1, C) for i in 1:b:C)
minibatch(x, batchsize) = Any[cview(x, ind) for ind in indbatch(x, batchsize)]
minibatch(x, y, batchsize) = Any[(cview(x, ind), cview(y, ind)) for ind in indbatch(x, batchsize)]