__precompile__(true)

module FluxUtils

using Printf, LinearAlgebra, Statistics, Distributed, Random
using Flux, BSON, ProgressMeter, Parameters, Requires

using Flux: glorot_uniform, gate, zeros, ones, stack, unsqueeze, chunk
using Flux: param, mapleaves, reset!, loadparams!
using Flux: RNNCell, LSTMCell, GRUCell, Recur, OneHotMatrix, OneHotVector
import Flux: hidden, @treelike, testmode!

using Flux.Optimise: update!, apply!, train!
using Flux.Optimise: Optimiser, Params, WeightDecay, ϵ, runall, IdDict

include("math.jl")
include("batch.jl")
include("layer.jl")
include("recurrent.jl")
include("util.jl")
include("params.jl")
include("vector.jl")
include("sklearn.jl")
include("train.jl")
include("optimizer.jl")
if isdefined(Flux, :CuArrays)
    include("cuda.jl")
end

function __init__()
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
end

end
