__precompile__(true)

module FluxUtils

using Flux, BSON, Adapt, Utils, Requires, Suppressor

include("math.jl")
include("batch.jl")
include("layer.jl")
include("flstm.jl")
include("fix.jl")
include("convert.jl")
include("params.jl")
include("vector.jl")
include("io.jl")
include("sklearn.jl")
include("loss.jl")

@init @suppress include(joinpath(@__DIR__, "optimizer.jl"))
@init @suppress include(joinpath(@__DIR__, "train.jl"))

if VERSION >= v"0.7"
    @init @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
    @init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("fixcu.jl")
else
    @require MPI include("mpi.jl")
    @require CuArrays include("fixcu.jl")
end

end