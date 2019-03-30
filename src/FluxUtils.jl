__precompile__(true)

module FluxUtils

using Flux, BSON, Adapt, Requires, Utils, ProgressMeter
using Flux: @treelike

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
include("train.jl")
include("optimizer.jl")

function __init__()
    @require MPI="da04e1cc-30fd-572f-bb4f-1f8673147195" include("mpi.jl")
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("fixcu.jl")
end

end
