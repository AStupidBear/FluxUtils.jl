# Sklearn Interface and Distributed Training for Flux.jl

## Installation

```julia
using Pkg
pkg"add FluxUtils"
```

## Usage

```julia
using Flux, FluxUtils
using FluxUtils: fit!, predict!
```

### Sklearn Interface for Time Series Prediction

First, define a simple LSTM network model.

```julia
model = Chain(LSTM(10, 10), Dense(10, 1)) |> gpu
```

Then, sepify the loss function for this model. `xs`/`ys` is a `Vector` of `AbstractArray`s of length `seqsize`.

```julia
loss = function (m, xs, ys)
    l, seqsize = 0f0, length(xs)
    for t in 1:seqsize
        x, y = xs[t], ys[t]
        l += mse(m(x), y)
    end
    return l / Float32(seqsize)
end
# The above is equivalent to 
# loss = seqloss(mse)
```

A `spec` is also need, which can be a NamedTuple, Dict or custom struct with at least three fileds defined.

```
spec = (epochs = 2, batchsize = 20, seqsize = 10)
```

Finally, create the optimizer `opt` and estimator `est`.

```julia
opt = ADAMW(1f-3)
est = Estimator(model, loss, opt, spec)
```

You can use `fit!` to fit this estimator just like `fit` in sklearn, with minibatching, logging, parameter syncronization, callbacks all handled internally. `fit!` will first create an minibatch sequence iterator from `x` and `y` with batch size `@unpack batchsize = spec` and sequence length `@unpack seqsize = spec` (truncated backpropagation).

```julia
F = 10     # feature dimension
N = 10000  # batch dimension
T = 1000   # time dimension
x = zeros(Float32, F, N, T)
y = zeros(Float32, 1, N, T)
```

```julia
fit!(est, x, y)
```

After the model is trained, you can use `predict!` to fill in the preallocated `ŷ` with predictions of `est` on `x` (because it's difficult to infer the output shape of a model wihtout running it).

```julia
ŷ = fill!(similar(y), 0)
predict!(ŷ, est, x)
```

Note that the type of `x`, `y` or `ŷ` is not restricted to AbstractArray, it can be Tuples of AbstractArrays. This is similar to the notion of multiple inputs and outputs in Keras.

If you are not dealing with time series problems, just add a dummy time dimension to your data. If your input is multidimensional, for example `size(x) == (W, H, C, N, T)`, you can reshape it to be of three dimensions `(F, N, T)` and reshape back in the definition of `m(x)` like this

```julia
function (m::MyModel)(x)
    x = reshape(x, W, H, C, N)
    ...
end
```

### Distributed Training with MPI

Distributed training can be achived with MPI with just a couple lines of code needed to be added. `fit!` internally will intercept `Flux`'s parameter updating step, apply `Allreduce` to average gradients from diffrent processes, then continue the updating. It will also synchronize parameters by broadcasting parameters from rank 0 to the rest before backpropagation starts.

If you want to train on NVIDIA GPUs, make sure you have built MPI with CUDA support (see [link](https://www.open-mpi.org/faq/?category=buildcuda)).

A template may be like this (run with `mpirun -np 4 julia *`)

```julia
using MPI
MPI.Init()
# ... code to load data
# ... code to define est
fit!(est, x, y)
# ... code to predict
MPI.Finalize()
```

The complete example is located at `test/mpi.jl`.

### Dealing with Big Data

Because data is only lazily subsliced in the training process, you can use memory mapping to read large datasets. [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) is recommended for this kind of usage. The function `h5concat` in [HDF5Utils.jl](https://github.com/AStupidBear/HDF5Utils.jl) can help you concatenate a large amount of files into a single file efficiently.


```julia
using HDF5
x, y = open("data.h5", "r") do fid
    readmmap(fid["x"]), 
    readmmap(fid["y"])
end
```

### Utility Functions

```julia
m = Chain(LSTM(10, 10), Dense(10, 1))
```

untrack all tracked objects in m

```julia
notrack(m)
```

concatenate all parameters of a model to a single vector

```julia
v = net2vec(m)
```

copy `v` to parameters of `m`

```julia
vec2net!(m, v)
```

concatenate all gradients of a model to a single vector

```julia
net2grad(m)
```

get all parameters of a model with names

```julia
namedparams(m)
```

get all states of a model

```julia
s = states(m)
```

load states `s` back into model `m`

```julia
loadstates!(m, s)
```

get all weights of a model (without biases), useful for regularization

```julia
weights(m)
```

batch matrix-matrix product (can be differentiated)

```julia
bmm(A, B)
```