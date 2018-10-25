softσ(x) = x / (one(x) + abs(x)) / oftype(x, 2) - oftype(x, 0.5)

pσ(x) = (x = x / 4.1f0; ifelse(x > 1f0, 1f0, ifelse(x < -1f0, 0f0, 0.5f0 + x * (1f0 - abs(x) / 2f0))))

function ptanh(x)
    xθ, yθ, λ = 1.92033f0, 0.96016f0, 0.26037f0
    ifelse(x > xθ, yθ, ifelse(x > 0f0, yθ - λ * (x - xθ)^2, ifelse(x > -xθ, λ * (x + xθ)^2 - yθ, -yθ)))
end