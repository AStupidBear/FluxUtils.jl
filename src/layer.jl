export DenseReLU, GaussianNoise

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

mutable struct GaussianNoise{F}
  stddev::F
  active::Bool
end

GaussianNoise(stddev) = GaussianNoise(stddev, true)

function (a::GaussianNoise)(x)
  a.active || return x
  y = rand!(similar(x))
  return x .+ y
end

Flux._testmode!(a::GaussianNoise, test) = (a.active = !test)