DenseReLU(in, out; ka...) = Dense(in, out, relu; ka...)

function Flux.RNN(in::Integer, out::Integer, nlayers::Integer, 
            dropout = 0f0, recurrent = true, layer = LSTM; ka...) 
    layers = []
    layer = !recurrent ? DenseReLU : layer
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