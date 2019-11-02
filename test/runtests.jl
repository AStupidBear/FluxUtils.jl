using Random, Statistics
using Flux, FluxUtils, Test
using FluxUtils: fit!, predict!

F, N, T = 10, 1000, 1000
x = randn(Float32, F, N, T)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 10), Dense(10, 1)) |> gpu
loss = seqloss(Flux.mse)
spec = (epochs = 2, batchsize = 100, seqsize = 100)
opt = ADAMW32(1f-3)
est = Estimator(model, loss, opt, spec)
fit!(est, x, y)

ŷ = fill!(similar(y), 0)
predict!(ŷ, est, x)
@test Flux.mse(y, ŷ) < 0.01

notrack(est)
v = net2vec(est)
vec2net!(est, v)
net2grad(est)
namedparams(est)
s = states(est)
loadstates!(est, s)
weights(est)
A = zeros(10, 2, 10) |> gpu
B = zeros(2, 3, 10) |> gpu
bmm(A, B)

A = rand(2, 3) |> gpu
B = Flux.onehotbatch([:b, :a], [:a, :b, :c]) |> gpu
@test A * B == A * Array(B) 

if "MPI" in keys(Pkg.installed())
    using MPI: mpiexec
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    file = joinpath(@__DIR__, "mpi.jl")
    run(`$mpiexec -np 4 $julia $file`)
end