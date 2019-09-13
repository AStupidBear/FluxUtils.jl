export DenseReLU, GaussianNoise, WinsorNorm

DenseReLU(in, out; ka...) = Dense(in, out, relu; ka...)

function Flux.RNN(in::Integer, out::Integer, nlayers::Integer, 
                dropout = 0f0, layer = LSTM; ka...) 
    layers = []
    for n in 1:nlayers
        if n == 1
            push!(layers, layer(in, out; ka...))
        else
            dropout > 0 && push!(layers, Dropout(Float32(dropout)))
            push!(layers, layer(out, out; ka...))
        end
    end
    Chain(layers...)
end

# GaussianNoise
mutable struct GaussianNoise{F}
  stddev::F
  active::Bool
end

GaussianNoise(stddev) = GaussianNoise(stddev, true)

function (a::GaussianNoise)(x)
  a.active || return x
  a.stddev == 0 && return x
  y = lmul!(a.stddev, randn!(similar(x)))
  return x .+ y
end

Flux._testmode!(a::GaussianNoise, test) = (a.active = !test)

# WinsorNorm
mutable struct WinsorNorm
    θd::Vector{Float32}
    θu::Vector{Float32}
    μ::Vector{Float32}
    σ::Vector{Float32}
end

WinsorNorm() = WinsorNorm(zeros(Float32, 0), zeros(Float32, 0), zeros(Float32, 0), zeros(Float32, 0))

function (m::WinsorNorm)(x)
    @unpack θd, θu, μ, σ = m
    @. ifelse(isnan(x), μ, (clamp(x, θd, θu) - μ) / σ)
end

function fit!(m::WinsorNorm, x)
    x = reshape(x, size(x, 1), :)
    F, N = size(x)
    θd = zeros(Float32, F)
    θu = zeros(Float32, F)
    μ = zeros(Float32, F)
    σ = zeros(Float32, F)
    @showprogress "winsorize..." for f in 1:F
        y = filter(!isnan, x[f, :])
        if !isempty(y)
            θd[f], θu[f] = quantile(y, [0.01f0, 0.99f0])
            clamp!(y, θd[f], θu[f])
            μ[f], σ[f] = mean(y), std(y)
            σ[f] = ifelse(isnan(σ[f]), 1f0, σ[f])
        else
            θd[f] = θu[f] = μ[f] = 0f0
            σ[f] = 1f0
        end
    end
    @pack! m = θd, θu, μ, σ
end