export forwardmode, float32, float64, gpu32

Flux.adapt(::Type{<:Array{T}}, xs::AbstractArray) where T <: AbstractFloat = convert(Array{T}, xs)
Flux.mapleaves(f, x::AbstractArray{<:Number}) = f(x)

float32(m) = mapleaves(x -> Flux.adapt(Array{Float32}, x), m)
float64(m) = mapleaves(x -> Flux.adapt(Array{Float64}, x), m)
gpu32(m) = gpu(float32(m))

forwardmode(m) = mapleaves(Flux.data, m)