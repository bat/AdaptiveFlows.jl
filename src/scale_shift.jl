# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

struct ScaleShiftModule <: AbstractFlowModule
    A::Matrix{Real}
    b::Vector{Real}
end

export ScaleShiftModule
@functor ScaleShiftModule

function ScaleShiftModule(stds::AbstractVector, means::AbstractVector)
    A = Diagonal(inv.(stds))
    return ScaleShiftModule(A, .- A * means)
end

function ScaleShiftModule(x::AbstractArray)
    stds = vec(std(x, dims = 2))
    means = vec(mean(x, dims = 2))
    ScaleShiftModule(stds, means)
end

function ChangesOfVariables.with_logabsdet_jacobian(f::ScaleShiftModule, x::Any)
    y, ladj = ChangesOfVariables.with_logabsdet_jacobian(MulAdd(f.A, f.b), x)

    return y, fill(ladj, 1, size(y,2))
end

(f::ScaleShiftModule)(x::AbstractMatrix) = MulAdd(f.A, f.b)(x)
(f::ScaleShiftModule)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::ScaleShiftModule)
    A = inv(f.A)
    return ScaleShiftModule(A, .- A * f.b)
end
