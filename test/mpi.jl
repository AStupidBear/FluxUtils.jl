using Statistics, Random, Test
using Flux, FluxUtils, MPI
using FluxUtils: fit!, predict!

MPI.Init()

F, N, T = 10, 1000, 1000
Random.seed!(1234)
x = randn(Float32, F, N, T)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 10), Dense(10, 1))
loss = seqloss(Flux.mse)
spec = (epochs = 10, batchsize = 100, seqsize = 100)
opt = ADAMW32(1f-3)
est = Estimator(model, loss, opt, spec)

fit!(est, x, y)

ŷ = fill!(similar(y), 0)
predict!(ŷ, est, x)
@test Flux.mse(y, ŷ) < 0.01

MPI.Finalize()