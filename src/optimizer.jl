export Clip, ADAMW32

mutable struct Clip
    thresh::Float32
end

function Flux.Optimise.apply!(o::Clip, x, Δ)
    θ = o.thresh
    @. Δ = clamp(Δ, -θ, θ)
    return Δ
end

mutable struct ADAM32
    eta::Float32
    beta::Tuple{Float32, Float32}
    state::IdDict
end

ADAM32(η = 0.001, β = (0.9, 0.999)) = ADAM32(η, β, IdDict())

function Flux.Optimise.apply!(o::ADAM32, x, Δ)
    η, β = o.eta, o.beta
    mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    o.state[x] = (mt, vt, βp .* β)
    return Δ
end

ADAMW32(η = 0.001, decay = 1f-3, θ = 0.5f0) =
    Optimiser(Clip(θ), ADAM32(η), WeightDecay(decay))
