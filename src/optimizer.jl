using Flux.Optimise: optimiser, invdecay, descent, momentum, rmsprop, adam

@init @suppress begin

Flux.Optimise.SGD(ps, η = 0.1f0; decay = 0) =
  optimiser(ps, p -> invdecay(p, decay), p -> descent(p,η))

Flux.Optimise.Momentum(ps, η = 1f-2; ρ = 0.9f0, decay = 0) =
  optimiser(ps, p -> invdecay(p,decay), p -> momentum(p, ρ, η), p -> descent(p,1))

Flux.Optimise.RMSProp(ps, η = 1f-3; ρ = 0.9f0, ϵ = 1f-8, decay = 0) =
  optimiser(ps, p -> rmsprop(p; η = η, ρ = ρ, ϵ = ϵ), 
          p -> invdecay(p, decay), p -> descent(p, 1))

Flux.Optimise.ADAM(ps, η = 1f-3; β1 = 0.9f0, β2 = 0.999f0, ϵ = 1f-08, decay = 0) =
  optimiser(ps, p -> adam(p; η = η, β1 = β1, β2 = β2, ϵ = ϵ), 
            p -> invdecay(p, decay), p -> descent(p, 1))

end