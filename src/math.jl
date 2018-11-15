export pσ, ptanh, ∇pσ, ∇ptanh

softσ(x) = x / (one(x) + abs(x)) / oftype(x, 2) - oftype(x, 0.5)

function pσ(x)
    l, o, h = one(x), zero(x), oftype(x, 0.5)
    x = x / oftype(x, 4.1)
    ifelse(x > l, l, ifelse(x < -l, o, h + x * (l - h * abs(x))))
end

function ∇pσ(x)
    l, o, h = one(x), zero(x), oftype(x, 0.5)
    x = x / oftype(x, 4.1)
    x * (l - h * abs(x))
    ifelse(x > l, o, ifelse(x < -l, o, l - abs(x)))
end

function ptanh(x)
    l, o = one(x), zero(x)
    xθ = oftype(x, 1.92033)
    yθ = oftype(x,  0.96016)
    λ = oftype(x, 0.26037)
    ifelse(x > xθ, yθ, ifelse(x > o, yθ - λ * (x - xθ)^2, ifelse(x > -xθ, λ * (x + xθ)^2 - yθ, -yθ)))
end

function ∇ptanh(x)
    l, o = one(x), zero(x)
    xθ = oftype(x, 1.92033)
    yθ = oftype(x,  0.96016)
    λ2 = oftype(x, 2 * 0.26037)
    ifelse(x > xθ, o, ifelse(x > o, -λ2 * (x - xθ), ifelse(x > -xθ, λ2 * (x + xθ), o)))
end
