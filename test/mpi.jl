using Test
using Statistics
using Flux
using FluxUtils
using FluxUtils: fit!, predict!

using MPIClusterManagers
const MCM = MPIClusterManagers

F, N, T = 10, 1000, 1000
x = randn(Float32, F, N, T)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 10), Dense(10, 1))
loss = seqloss(Flux.mse)
spec = (epochs = 10, batchsize = 100, seqsize = 100)
opt = ADAMW32(1f-3)
est = Estimator(model, loss, opt, spec)

man = MCM.start_main_loop(MCM.MPI_TRANSPORT_ALL)

@mpi_do man fit!(est, x, y)

ŷ = fill!(similar(y), 0)
predict!(ŷ, est, x)
@test Flux.mse(y, ŷ) < 0.01

MCM.stop_main_loop(man)