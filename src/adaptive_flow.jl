# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

abstract type AbstractFlow <: Function
end


struct CompositeFlow <: AbstractFlow
    flow::Function
end

export CompositeFlow
@functor CompositeFlow

function CompositeFlow(modules::Vector{F}) where F <: Function
    return CompositeFlow(fchain(modules))
end    

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CompositeFlow,
    x::Any
)
    with_logabsdet_jacobian(f.flow, x)
end

(f::CompositeFlow)(x::Any) = f.flow(x)
(f::CompositeFlow)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CompositeFlow)
    return CompositeFlow(InverseFunctions.inverse(f.flow).fs)
end


abstract type AbstractFlowModule <: AbstractFlow
end


abstract type AbstractFlowBlock <: AbstractFlowModule
end
