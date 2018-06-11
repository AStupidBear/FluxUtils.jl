module FluxUtils

using Flux, Utils

export indbatch, minibatch, indbatchseq, minibatchseq
export MLP

include("batch.jl")
include("layer.jl")

end