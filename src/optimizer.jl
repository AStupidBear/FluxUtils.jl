using Flux.Optimise: Param

@init @suppress begin

function Flux.Optimise.descent(p::Param, η::Float32)
  function ()
    @. p.x -= η * p.Δ
    @. p.Δ = 0
  end
end

function Flux.Optimise.momentum(p::Param, ρ, η)
  v = zeros(p.x)
  function ()
    @. v = ρ * v - η * p.Δ
    @. p.Δ = -v
  end
end

# Ref. https://arxiv.org/pdf/1212.0901.pdf
function Flux.Optimise.nesterov(p::Param, ρ, η)
  v = zeros(p.x)
  function ()
    d = @. ρ^2 * v - (1+ρ) * η * p.Δ
    @. v = ρ*v - η*p.Δ
    @. p.Δ = -d
  end
end

function Flux.Optimise.rmsprop(p::Param; η::Float32 = 0.001f0, ρ::Float32 = 0.9f0, ϵ::Float32 = 1f-8)
  acc  = zeros(p.x)
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ^2
    @. p.Δ *= η / √(acc + ϵ)
  end
end

function Flux.Optimise.adagrad(p::Param; η::Float32 = 0.01f0, ϵ::Float32 = 1f-8)
  acc = zeros(p.x) .+ ϵ
  function ()
    @. acc += p.Δ^2
    @. p.Δ *= η / √(acc + ϵ)
  end
end

function Flux.Optimise.adadelta(p::Param; ρ::Float32 = 0.9f0, ϵ::Float32 = 1f-8)
  acc = zeros(p.x)
  Δacc = zeros(p.x)
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ^2
    @. p.Δ *= √(Δacc + ϵ) / √(acc + ϵ)
    @. Δacc = ρ * Δacc + (1 - ρ) * p.Δ^2
   end
end

function Flux.Optimise.adam(p::Param; η::Float32 = 0.001f0, β1::Float32 = 0.9f0, β2::Float32 = 0.999f0, ϵ::Float32 = 1f-8)
  mt = zeros(p.x)
  vt = zeros(p.x)
  β1p, β2p = β1, β2
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ^2
    @. p.Δ =  mt / (1 - β1p) / √(vt / (1 - β2p) + ϵ) * η
    β1p *= β1
    β2p *= β2
  end
end

function Flux.Optimise.adamax(p::Param; η::Float32 = 0.002f0, β1::Float32 = 0.9f0, β2::Float32 = 0.999f0, ϵ::Float32 = 1f-8)
  mt = zeros(p.x)
  ut = zeros(p.x)
  β1p = β1
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. ut = max(β2 * ut, abs(p.Δ))
    @. p.Δ = (η/(1 - β1p)) * mt/(ut + ϵ)
    β1p *= β1
  end
end

function Flux.Optimise.amsgrad(p::Param; η::Float32 = 0.001f0, β1::Float32 = 0.9f0, β2::Float32 = 0.999f0, ϵ::Float32 = 1f-8)
  mt = zeros(p.x)
  vt = zeros(p.x) .+ ϵ
  v̂t = zeros(p.x) .+ ϵ
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ ^ 2
    @. v̂t = max.(v̂t, vt)
    @. p.Δ = η * mt / √v̂t
  end
end

function Flux.Optimise.nadam(p::Param; η::Float32 = 0.001f0, β1::Float32 = 0.9f0, β2::Float32 = 0.999f0, ϵ::Float32 = 1f-8)
  mt = zeros(p.x)
  vt = zeros(p.x)
  β1p, β2p = β1, β2
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ^2
    @. p.Δ = (β1 * mt / (1 - β1 * β1p) + (1 - β1) * p.Δ / (1 - β1p)) / √(vt * β2 / (1 - β2p) + ϵ) * η
    β1p *= β1
    β2p *= β2
  end
end

Flux.Optimise.clip(p::Param, thresh::Float32) = () -> clamp!(p.Δ, -thresh, thresh)

function Flux.Optimise.expdecay(p::Param, γ::Float32)
  if γ != 0
    return () -> p.Δ .+= γ .* p.x
  else
    return () -> nothing
  end
end

function Flux.Optimise.invdecay(p::Param, γ::Float32)
  if γ != 0
    n = 0
    return () -> begin
      p.Δ .*= 1 / (1 + γ * n)
      n += 1
    end
  else
    return () -> nothing
  end
end

end