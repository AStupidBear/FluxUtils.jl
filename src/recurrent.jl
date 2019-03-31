using Flux: glorot_uniform, param, gate, Recur, TrackedArray
using Flux: zeros, ones
import Flux: hidden

export FLSTM, SGRU, MGU, SMGU, hBatch

hBatch(x::AbstractVector, h::AbstractVector) = h
hBatch(x::AbstractMatrix, h::AbstractVector{T}) where T = repeat(h, 1, size(x, 2))
hBatch(x::AbstractMatrix, h::AbstractMatrix{T}) where T =
    size(h, 2) == size(x, 2) ? h : repeat(h[:, 1], 1, size(x, 2))

function (a::Dense{<:Any, <:TrackedArray})(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W * x +ᵇ b)
end

function (a::Dense{<:Any, <:Array})(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    y = W * x
    y .= σ.(y .+ b)
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
    cell = FLSTMCell(param(init(4out, in)), param(init(4out, out)), param(init(4out)),
                    param(zeros(out)), param(zeros(out)))
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

function (m::FLSTMCell{<:Array})(h_, x)
    h, c = hBatch(x, h_[1]), hBatch(x, h_[2])
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    for j in 1:size(h, 2), i in 1:size(h, 1)
        forget = pσ(gx[i + o, j] + gh[i + o, j] + b[i + o])
        c[i, j] *= forget
    end
    for j in 1:size(h, 2), i in 1:size(h, 1)
        input = pσ(gx[i, j] + gh[i, j] + b[i])
        cell = ptanh(gx[i + 2o, j] + gh[i + 2o, j] + b[i + 2o])
        c[i, j] += input * cell
    end
    for j in 1:size(h, 2), i in 1:size(h, 1)
        output = pσ(gx[i + 3o, j] + gh[i + 3o, j] + b[i + 3o])
        h[i, j] = output * ptanh(c[i, j])
    end
    return (h, c), h
end

hidden(m::FLSTMCell) = (m.h, m.c)

@treelike FLSTMCell

namedchildren(m::FLSTMCell) = zip(fieldnames(typeof(m)), children(m))

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
            param(zeros(3out)), param(zeros(out)))

function (m::SGRUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = pσ.(gate(gh, o, 1) +ᵇ gate(b, o, 1))
    z = pσ.(gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h̃ = ptanh.(gx +ᵇ r *ᵇ gate(gh, o, 3) +ᵇ gate(b, o, 3))
    h′ = (1f0 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::SGRUCell{<:Array})(h, x)
    h = hBatch(x, h)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    for j in 1:size(h, 2), i in 1:size(h, 1)
        r = pσ(gh[i, j] + b[i])
        z = pσ(gh[i + o, j] + b[i + o])
        h̃ = ptanh(gx[i, j] + r * gh[i + 2o, j] + b[i + 2o])
        h[i, j] = (1f0 - z) * h̃ + z * h[i, j]
    end
    return h, h
end

hidden(m::SGRUCell) = m.h

@treelike SGRUCell

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
            param(zeros(2out)), param(zeros(out)))

function (m::MGUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = pσ.(gate(gx, o, 1) +ᵇ gate(gh, o, 1) +ᵇ gate(b, o, 1))
    h̃ = ptanh.(gate(gx, o, 2) +ᵇ r *ᵇ gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h′ = (1f0 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::MGUCell{<:Array})(h, x)
    h = hBatch(x, h)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    for j in 1:size(h, 2), i in 1:size(h, 1)
        r = z = pσ(gx[i, j] + gh[i, j] + b[i])
        h̃ = ptanh(gx[i + o, j] + r * gh[i + o, j] + b[i + o])
        h[i, j] = (1f0 - z) * h̃ + z * h[i, j]
    end
    return h, h
end

hidden(m::MGUCell) = m.h

@treelike MGUCell

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
            param(zeros(2out)), param(zeros(out)))

function (m::SMGUCell{<:TrackedArray})(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    r = z = pσ.(gate(gh, o, 1) +ᵇ gate(b, o, 1))
    h̃ = ptanh.(gx +ᵇ r *ᵇ gate(gh, o, 2) +ᵇ gate(b, o, 2))
    h′ = (1f0 -ᵇ z) *ᵇ h̃ +ᵇ z *ᵇ h
    return h′, h′
end

function (m::SMGUCell{<:Array})(h, x)
    h = hBatch(x, h)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi * x, m.Wh * h
    for j in 1:size(h, 2), i in 1:size(h, 1)
        r = z = pσ(gh[i, j] + b[i])
        h̃ = ptanh(gx[i, j] + r * gh[i + o, j] + b[i + o])
        h[i, j] = (1f0 - z) * h̃ + z * h[i, j]
    end
    return h, h
end

hidden(m::SMGUCell) = m.h

@treelike SMGUCell

Base.show(io::IO, l::SMGUCell) =
    print(io, "SMGUCell(", size(l.Wi, 2), ", ", size(l.Wh, 1) ÷ 2, ")")

SMGU(a...; ka...) = Recur(SMGUCell(a...; ka...))

namedchildren(m::Union{FLSTMCell, SGRUCell, MGUCell, SMGUCell}) = zip(fieldnames(typeof(m)), children(m))

# using Flux: data
# using Flux.Tracker: track, @grad

# istrain(m, args...) = any(x -> isa(x, TrackedArray), (m.Wi, m.Wh, m.b, args...))

# function forward!(m::SMGUCell, h, x, Wi, Wh, b)
#     h = hBatch(x, h)
#     o = size(h, 1)
#     gx, gh = Wi * x, Wh * h
#     for j in 1:size(h, 2), i in 1:size(h, 1)
#         u = gh[i, j] + b[i]
#         r = z = pσ(u)
#         v = gx[i, j] + r * gh[i + o, j] + b[i + o]
#         h̃ = ptanh(v)
#         h[i, j] = (1f0 - z) * h̃ + z * h[i, j]
#     end
#     return h, h
# end

# function (m::SMGUCell)(h, x)
#     result = istrain(m, h, x) ?
#     track(m, h, x, m.Wi, m.Wh, m.b) :
#     Flux.Tracker._forward(m, h, x, m.Wi, m.Wh, m.b)[1]
#     # forward!(m, h, x, m.Wi, m.Wh, m.b)
#     result[1], result[2]
# end

# @grad function (m::SMGUCell)(h̄, x̄, W̄i, W̄h, b̄)
#     h, x, Wi, Wh, b = data(h̄), data(x̄), data(W̄i), data(W̄h), data(b̄)
#     h = hBatch(x, h)
#     o = size(h, 1)
#     gx, gh = Wi * x, Wh * h
#     r = z = zero(h)
#     u, v, h̃, h′ = zero(h), zero(h), zero(h), zero(h)
#     for j in 1:size(h, 2), i in 1:size(h, 1)
#         u[i, j] = gh[i, j] + b[i]
#         z[i, j] = pσ(u[i, j])
#     end
#     for j in 1:size(h, 2), i in 1:size(h, 1)
#         v[i, j] = gx[i, j] + r[i, j] * gh[i + o, j] + b[i + o]
#         h̃[i, j] = ptanh(v[i, j])
#     end
#     for j in 1:size(h, 2), i in 1:size(h, 1)
#         h′[i, j] = (1f0 - z[i, j]) * h̃[i, j] + z[i, j] * h[i, j]
#     end
#     return (h′, h′), @closure Δ -> begin
#         @assert Δ[1] == 0
#         dh, db = zero(h), zero(b)
#         dgx, dgh = gx, gh
#         dh̃, dz = zero(h̃), zero(z)
#         for j in 1:size(h, 2), i in 1:size(h, 1)
#             # δ = Δ[1][i, j] + Δ[2][i, j]
#             δ = Δ[2][i, j]
#             dh[i, j] += δ * z[i, j]
#             dh̃[i, j] = δ * (1f0 - z[i, j])
#             dz[i, j] = δ * (h[i, j] - h̃[i, j])
#         end
#         for j in 1:size(h, 2), i in 1:size(h, 1)
#             dv = dh̃[i, j] * (1f0 - v[i, j]^2)                  # ∇ptanh(v[i, j])
#             dgx[i, j] += dv
#             dgh[i + o] += dv * r[i, j]
#             db[i + o] += dv
#         end
#         for j in 1:size(h, 2), i in 1:size(h, 1)
#             du = dz[i, j] * (1f0 - u[i, j]) * u[i, j]           # ∇pσ(u[i, j])
#             dgh[i, j] += du
#             db[i] += du
#         end
#         BLAS.gemm!('T', 'N', oneel(dh), Wh, dgh, oneel(dh), dh)  # dh .+= transpose(Wh) * dgh
#         dx = transpose(Wi) * dgx
#         dWi = dgx * transpose(dx)
#         dWh = dgh * transpose(dh)
#         dh = ndims(data(h̄)) > 1 ? dh : vec(sum(dh, dims = 2))
#         return dh, dx, dWi, dWh, db
#     end
# end
