export wmse

wmse(ŷ, y, w) = sum(w .* (ŷ .- y).^2) / length(y)
