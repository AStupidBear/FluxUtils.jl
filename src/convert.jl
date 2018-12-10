using Flux.Tracker: istracked

export notrack, float32, float64, gpu32

if isdefined(Adapt, :adapt_structure)
    Adapt.adapt_structure(::Type{<:Array{T}}, xs::AbstractArray) where T <: AbstractFloat = convert(Array{T}, xs)
elseif isdefined(Adapt, :adapt_)
    Adapt.adapt_(::Type{<:Array{T}}, xs::AbstractArray) where T <: AbstractFloat = convert(Array{T}, xs)
end
Flux.mapleaves(f, x::AbstractArray{<:Number}) = f(x)

float32(m) = mapleaves(x -> Flux.adapt(Array{Float32}, x), m)
float64(m) = mapleaves(x -> Flux.adapt(Array{Float64}, x), m)
gpu32(m) = gpu(float32(m))

function notrack(m)
    keep = Ref(true)
    Flux.prefor(p -> istracked(p) && (keep[] = false), m)
    keep[] ? m : mapleaves(Flux.data, m)
end
