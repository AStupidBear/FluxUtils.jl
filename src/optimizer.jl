using Flux.Optimise: IdDict, Optimiser, WeightDecay, ϵ
import Flux.Optimise: apply!

export ADAMW32

mutable struct ADAM32
    eta::Float32
    beta::Tuple{Float32, Float32}
    state::IdDict
end

ADAM32(η = 0.001, β = (0.9, 0.999)) = ADAM32(η, β, IdDict())

function apply!(o::ADAM32, x, Δ)
    η, β = o.eta, o.beta
    mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    o.state[x] = (mt, vt, βp .* β)
    return Δ
end

ADAMW32(η = 0.001, decay = 0) =
    Optimiser(ADAM32(η), WeightDecay(decay))
