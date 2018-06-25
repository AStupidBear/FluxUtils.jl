using Flux: glorot_uniform, param, initn, gate, treelike, Recur

mutable struct FLSTMCell{A,V}
    Wi::A
    Wh::A
    b::V
    h::V
    c::V
end

function FLSTMCell(in::Integer, out::Integer; init = glorot_uniform)
  cell = FLSTMCell(param(init(out*4, in)), param(init(out*4, out)), param(zeros(out*4)),
                  param(initn(out)), param(initn(out)))
  cell.b.data[gate(out, 2)] = 1
  return cell
end

function (m::FLSTMCell)(h_, x)
    h, c = h_ # TODO: nicer syntax on 0.7
    b, o = m.b, size(h, 1)
    g = m.Wi*x .+ m.Wh*h .+ b
    input = softσ.(gate(g, o, 1))
    forget = softσ.(gate(g, o, 2))
    cell = softsign.(gate(g, o, 3))
    output = softσ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* softsign.(c)
    return (h′, c), h′
end

Flux.hidden(m::FLSTMCell) = (m.h, m.c)

treelike(FLSTMCell)

namedchildren(m::FLSTMCell) = zip(fieldnames(m), children(m))

Base.show(io::IO, l::FLSTMCell) =
    print(io, "FLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

FLSTM(a...; ka...) = Recur(FLSTMCell(a...; ka...))

# function (m::FLSTMCell)(h_, x)
#     h, c = h_ # TODO: nicer syntax on 0.7
#     b, o = m.b, size(h, 1)
#     g = m.Wi*x .+ m.Wh*h .+ b
#     input = gate(g, o, 1)
#     forget = gate(g, o, 2)
#     cell = gate(g, o, 3)
#     output = gate(g, o, 4)
#     c = softσ.(forget) .* c .+ softσ.(input) .* softσ.(cell)
#     h′ = softσ.(output) .* softsign.(c)
#     return (h′, c), h′
# end