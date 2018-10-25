__precompile__(true)

module FluxUtils

using Compat, Compat.Printf, Compat.Distributed, Compat.LinearAlgebra
using Compat: axes, rmul!

using Flux, BSON, Adapt, Requires, Utils

macro treelike(ex)
    @static if VERSION >= v"0.7"
        esc(:(Flux.@treelike($ex)))
    else
        esc(:(Flux.treelike($ex)))
    end
end

include("math.jl")
include("batch.jl")
include("layer.jl")
include("recurrent.jl")
include("fix.jl")
include("convert.jl")
include("params.jl")
include("vector.jl")
include("io.jl")
include("sklearn.jl")
include("loss.jl")
include("broadcast.jl")

@static if VERSION >= v"0.7"
    include("optimizer.jl")
    include("train.jl")
    function __init__()
        @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
        @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("fixcu.jl")
    end
else
    using Suppressor
    @require Flux @suppress include("optimizer.jl")
    @require Flux @suppress include("train.jl")
    @require MPI include("mpi.jl")
    @require CuArrays include("fixcu.jl")
end

end