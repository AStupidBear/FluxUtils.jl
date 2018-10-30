using Flux: glorot_uniform, param, initn, gate, Recur, NNlib.@fix, TrackedArray
import Flux: hidden

export FLSTM, SGRU, MGU, SMGU

function (a::Dense{<:Any, <:TrackedArray})(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W * x +ᵇ b)
end

function (a::Dense{<:Any, <:Array})(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    y = W * x
    for j in 1:size(y, 2), i in 1:size(y, 1)
        y[i, j] = σ(y[i, j] + b[i])
    end
    return y
end

# FLSTM

mutable struct FLSTMCell{A, V}
    Wi::A
    Wh::A
    b::V
    h::V
    c::V
end

function FLSTMCell(in::Integer, out::Integer; init = glorot_uniform)
    cell = FLSTMCell(param(init(4out, in)), param(init(4out, out)), param(zeros(4out)),
                    param(initn(out)), param(initn(out)))
    cell.b.data[gate(out, 2)] .= 1
    return cell
end

function (m::FLSTMCell{<:TrackedArray})(h_, x)
    h, c = h_ # TODO: nicer syntax on 0.7
    b, o = m.b, size(h, 1)
    g = m.Wi * x +ᵇ m.Wh * h +ᵇ b
    input = pσ.(gate(g, o, 1))
    forget = pσ.(gate(g, o, 2))
    cell = ptanh.(gate(g, o, 3))
    output = pσ.(gate(g, o, 4))
    c = forget *ᵇ c +ᵇ input *ᵇ cell
    h′ = output *ᵇ ptanh.(c)
    return (h′, c), h′
end

function (m::FLSTMCell)(h_, x)
    h, c = h_ # TODO: nicer syntax on 0.7
    b, o = m.b, size(h, 1)
    g = m.Wi * x .+ m.Wh * h .+ b
    input = pσ.(gate(g, o, 1))
    forget = pσ.(gate(g, o, 2))
    cell = ptanh.(gate(g, o, 3))
    output = pσ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* ptanh.(c)
    return (h′, c), h′
end

hidden(m::FLSTMCell) = (m.h, m.c)

@treelike(FLSTMCell)

namedchildren(m::FLSTMCell) = zip(fieldnames(m), children(m))

Base.show(io::IO, l::FLSTMCell) =
    print(io, "FLSTMCell(", size(l.Wi, 2), ", ", size(l.Wh, 1) ÷ 4, ")")

FLSTM(a...; ka...) = Recur(FLSTMCell(a...; ka...))

# SGRU

mutable struct SGRUCell{A, V}
    Wi::A
    Wh::A
    b::V
    h::V
end

SGRUCell(in, out; init = glorot_uniform) =
    SGRUCell(param(init(out, in)), param(init(3out, out)),
            param(zeros(3out)), param(initn(out)))

function (m::SGRUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = @fix σ.(gate(gh, o, 1) +ᵇ gate(b, o, 1))
    z = @fix σ.(gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h̃ = @fix tanh.(gx +ᵇ r *ᵇ gate(gh, o, 3) +ᵇ gate(b, o, 3))
    h′ = (1 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::SGRUCell{<:Array})(h, x)
    if size(h, 2) == 1 && size(x, 2) > 1
        h = repeat(h, 1, size(x, 2))
    end
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = zeros(Float32, o, size(x, 2))
    z = zeros(Float32, o, size(x, 2))
    for j in 1:size(r, 2), i in size(r, 1)
        r[i, j] = pσ(gh[i, j] + b[i])
    end
    for j in 1:size(r, 2), i in size(r, 1)
        z[i, j] = pσ(gh[i + o, j] + gh[i + o, j] + b[i + o])
    end
    for j in 1:size(r, 2), i in size(r, 1)
        h′ = ptanh(gx[i, j] + r[i, j] * gh[i + 2o, j] + b[i + 2o])
        h[i, j] = (1f0 - z[i, j]) * h′ + z[i, j] * h[i, j]
    end
    return h, h
end

function (m::SGRUCell)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = pσ.(gate(gh, o, 1) .+ gate(b, o, 1))
    z = pσ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = ptanh.(gx .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    return h′, h′
end

hidden(m::SGRUCell) = m.h

@treelike(SGRUCell)

Base.show(io::IO, l::SGRUCell) =
    print(io, "SGRUCell(", size(l.Wi, 2), ", ", size(l.Wh, 1) ÷ 3, ")")

SGRU(a...; ka...) = Recur(SGRUCell(a...; ka...))

# MGU

mutable struct MGUCell{A, V}
    Wi::A
    Wh::A
    b::V
    h::V
end

MGUCell(in, out; init = glorot_uniform) =
    MGUCell(param(init(2out, in)), param(init(2out, out)),
            param(zeros(2out)), param(initn(out)))

function (m::MGUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = @fix σ.(gate(gx, o, 1) +ᵇ gate(gh, o, 1) +ᵇ gate(b, o, 1))
    h̃ = @fix tanh.(gate(gx, o, 2) +ᵇ r *ᵇ gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h′ = (1 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::MGUCell{<:Array})(h, x)
    if size(h, 2) == 1 && size(x, 2) > 1
        h = repeat(h, 1, size(x, 2))
    end
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = zeros(Float32, o, size(x, 2))
    for j in 1:size(r, 2), i in size(r, 1)
        r[i, j] = pσ(gx[i, j] + gh[i, j] + b[i])
    end
    for j in 1:size(r, 2), i in size(r, 1)
        h′ = ptanh(gx[i + o, j] + r[i, j] * gh[i + o, j] + b[i + o])
        h[i, j] = (1f0 - z[i, j]) * h′ + z[i, j] * h[i, j]
    end
    return h, h
end

function (m::MGUCell)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = pσ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    h̃ = ptanh.(gate(gx, o, 2) .+ r .* gate(gh, o, 2) .+ gate(b, o, 2))
    h′ = (1 .- z) .* h̃ .+ z .* h
    return h′, h′
end

hidden(m::MGUCell) = m.h

@treelike(MGUCell)

Base.show(io::IO, l::MGUCell) =
    print(io, "MGUCell(", size(l.Wi, 2), ", ", size(l.Wh, 1) ÷ 2, ")")

MGU(a...; ka...) = Recur(MGUCell(a...; ka...))

# SMGU

mutable struct SMGUCell{A, V}
    Wi::A
    Wh::A
    b::V
    h::V
end

SMGUCell(in, out; init = glorot_uniform) =
    SMGUCell(param(init(out, in)), param(init(2out, out)),
            param(zeros(2out)), param(initn(out)))

function (m::SMGUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = @fix σ.(gx +ᵇ gate(gh, o, 1) +ᵇ gate(b, o, 1))
    h̃ = @fix tanh.(gx +ᵇ r *ᵇ gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h′ = (1 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::SMGUCell{<:Array})(h, x)
    if size(h, 2) == 1 && size(x, 2) > 1
        h = repeat(h, 1, size(x, 2))
    end
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = zeros(Float32, o, size(x, 2))
    for j in 1:size(r, 2), i in size(r, 1)
        r[i, j] = pσ(gx[i, j] + gh[i, j] + b[i])
    end
    for j in 1:size(r, 2), i in size(r, 1)
        h′ = ptanh(gx[i, j] + r[i, j] * gh[i + o, j] + b[i + o])
        h[i, j] = (1f0 - z[i, j]) * h′ + z[i, j] * h[i, j]
    end
    return h, h
end

function (m::SMGUCell)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = pσ.(gx .+ gate(gh, o, 1) .+ gate(b, o, 1))
    h̃ = ptanh.(gx .+ r .* gate(gh, o, 2) .+ gate(b, o, 2))
    h′ = (1 .- z) .* h̃ .+ z .* h
    return h′, h′
end

hidden(m::SMGUCell) = m.h

@treelike(SMGUCell)

Base.show(io::IO, l::SMGUCell) = 
    print(io, "SMGUCell(", size(l.Wi, 2), ", ", size(l.Wh, 1) ÷ 2, ")")

SMGU(a...; ka...) = Recur(SMGUCell(a...; ka...))

namedchildren(m::Union{FLSTMCell, SGRUCell, MGUCell, SMGUCell}) = zip(fieldnames(typeof(m)), children(m))