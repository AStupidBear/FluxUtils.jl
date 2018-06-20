__precompile__(true)

module FluxUtils

using Flux, BSON, Adapt, Utils, Requires

export fσ, fsigmoid, ftanh, softσ, softsigmoid
export indbatch, minibatch, batchtupleseq
export FLSTM
export forwardmode, float32
export predict_seq, predict_batch
export namedparams
export weight_indices, net2vec, vec2net!
export savenet, loadnet!

include("math.jl")
include("batch.jl")
include("layer.jl")
include("flstm.jl")
include("fix.jl")
include("convert.jl")
include("device.jl")
include("predict.jl")
include("namedparams.jl")
include("vector.jl")
include("io.jl")

end