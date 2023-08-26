# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

"""
    AbstractFlow <: Function

Abstract supertype for all functions that are used as "Normalizing Flows" in this package. 
A Normalizing Flow is an invertible and diferentiable function from 
a `D`-dimensional space to a `D`-dimensional space. 
In this implementation, a flow may be applied to a batch of samples from a target distribution. 
Depending on the selected computing device and the specific flow, the input samples may be transformed in parallel.

Here, a flow returns a tuple of the transformed output of the flow, and a row matrix, with the `i`-th entry holding 
the logarithm of the absolute value of the determinant of the jacobian of the transformation of the `i`-th sample in 
the input.
"""
abstract type AbstractFlow <: Function
end

"""
    CompositeFlow <: AbstractFlow

A concrete subtype of `AbstractFlow`[@ref]. A `CompositeFlow` is a composition of several flow modules (see `AbstractFlowModule`[@ref]), 
individual normalizing flows, that each transform all components of the input data.
"""
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

"""
    AbstractFlowModule <: AbstractFlow

A concrete subtype of `AbstractFlow`[@ref]. Here, a flow module is a normalizing flow that transforms each of 
the input components. 
A flow module may consist of a scaling and shifting operation of the input samples, or be a composition of 
several flow blocks of a specific type (see `AbstractFlowBlock`). 
"""
abstract type AbstractFlowModule <: AbstractFlow
end

"""
    AbstractFlowBlock <: AbstractFlowModule

A concrete subtype of `AbstractFlowModule`[@ref]. Here, a flow block is a normalizing flow that may only 
transform a fraction of the components of the input samples. To transform all components of the input, 
several flow blocks must be composed to a flow module (see `AbstractFlowModule`).
"""
abstract type AbstractFlowBlock <: AbstractFlowModule
end
