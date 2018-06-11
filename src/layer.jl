function Flux.LSTM(in::Integer, out::Integer, nlayers::Integer, dropout = 0f0; ka...) 
    layers = []
    for n in 1:nlayers
        if n == 1
            push!(layers, LSTM(in, out; ka...))
        else
            push!(layers, Dropout(dropout))
            push!(layers, LSTM(out, out; ka...))
        end
    end
    Chain(layers...)
end

function MLP(in::Integer, out::Integer, nlayers::Integer, dropout = 0f0, σ = relu; ka...)
    layers = []
    for n in 1:nlayers
        if n == 1
            push!(layers, Dense(in, out, σ; ka...))
        else
            push!(layers, Dropout(dropout))
            push!(layers, Dense(out, out, σ; ka...))
        end
    end
    Chain(layers...)
end