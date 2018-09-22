export weightaxes, net2vec, vec2net!, net2grad

function weightaxes(m, maxnorm = false)
    pos, inds = 0, Vector{Int}[]
    for (name, p) in namedparams(m)
        if ndims(p) == 2
            @assert contains(lowercase(string(name)), "w")
            if maxnorm
                ind = eachrow(reshape(linearaxes(p) + pos, size(p)))
                append!(inds, ind)
            else
                push!(inds, linearaxes(p) + pos)
            end
        end
        pos += length(p)
    end
    return inds
end

parameters_to_vector(ps) = vcat(vec.(Flux.data.(ps))...)

function vector_to_parameters!(ps, x)
    pos = 1
    for p in ps
        pos_end = pos + length(p) - 1
        copy!(p.data, x[pos:pos_end])
        pos = pos_end + 1
    end
end

net2vec(m) = parameters_to_vector(Flux.params(m))

vec2net!(m, x) = vector_to_parameters!(Flux.params(m), x)

parameters_to_grad(ps) = vcat(vec.(Tracker.grad.(ps))...)

net2grad(m) = parameters_to_grad(Flux.params(m))