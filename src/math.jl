softσ(x) = x / (one(x) + abs(x)) / oftype(x, 2) - oftype(x, 0.5)

const softsigmoid = softσ

function fσ(x)
    x = x / oftype(x, 4.1)
    ifelse(x > 1, one(x), 
    ifelse(x < -1, zero(x),
    oftype(x, 0.5) + x * (one(x) - abs(x) / 2)))
end

const fsigmoid = fσ

function ftanh(x)
    xθ = oftype(x, 1.92033)
    yθ = oftype(x, 0.96016)
    λ = oftype(x, 0.26037)
    ifelse(x > xθ, yθ,
    ifelse(x > 0, yθ - λ * (x - xθ)^2, 
    ifelse(x > -xθ, λ * (x + xθ)^2 - yθ,
            -yθ)))
end