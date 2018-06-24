__precompile__(true)

module FluxUtils

using Flux, BSON, Adapt, Utils, Requires

export fσ, fsigmoid, ftanh, softσ, softsigmoid
export indbatch, minibatch, tupseqbatch
export FLSTM
export forwardmode, float32
export predseq, predseqbatch
export namedparams
export weightindices, net2vec, vec2net!
export savenet, loadnet!
export gradseq, gradbatch

include("math.jl")
include("batch.jl")
include("layer.jl")
include("flstm.jl")
include("fix.jl")
include("convert.jl")
include("predict.jl")
include("namedparams.jl")
include("vector.jl")
include("io.jl")
include("grad.jl")
include("device.jl")

end