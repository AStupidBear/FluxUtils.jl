DenseReLU(in, out; ka...) = Dense(in, out, relu; ka...)

function Flux.LSTM(in::Integer, out::Integer, nlayers::Integer, 
            dropout = 0f0, recurrent = true, fast = false; ka...) 
    layers = []
    layer = !recurrent ? DenseReLU : fast ? FLSTM : LSTM 
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