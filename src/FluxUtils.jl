module FluxUtils

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

end