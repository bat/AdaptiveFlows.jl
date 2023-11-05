# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

"""
    AbstractFlow <: Function

Abstract supertype for all functions that are used as "Normalizing Flows". 
A Normalizing Flow is an invertible and diferentiable function from 
a `D`-dimensional space to a `D`-dimensional space. 
A flow may be applied to a batch of samples from a target distribution. 
Depending on the selected computing device and the specific flow, the input samples may be transformed in parallel.

Here, a flow returns a tuple of the transformed output of the flow, and a row matrix, with the `i`-th entry holding 
the logarithm of the absolute value of the determinant of the jacobian of the transformation of the `i`-th sample in 
the input.
"""
abstract type AbstractFlow <: Function
end

"""
    CompositeFlow <: AbstractFlow

A `CompositeFlow` is a composition of several flow modules (see `AbstractFlowModule`[@ref]), 
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

function CompositeFlow(n_dims::Integer, modules::Vector{F}) where F <: Function
    build_flow(ndims, modules)
end    


function ChangesOfVariables.with_logabsdet_jacobian(
    f::F, 
    x::Matrix
) where F <: AbstractFlow
    with_logabsdet_jacobian(f.flow, x)
end

function ChangesOfVariables.with_logabsdet_jacobian(
    f::F,
    x::Vector
) where F <: AbstractFlow
    y, ladj = with_logabsdet_jacobian(f.flow, reshape(x,:,1))
    return vec(y), ladj[1]
end

(f::AbstractFlow)(x::Matrix) = f.flow(x)
(f::AbstractFlow)(x::Vector) = vec(f.flow(reshape(x, :, 1)))
(f::AbstractFlow)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CompositeFlow)
    return CompositeFlow(InverseFunctions.inverse(f.flow).fs)
end

"""
    prepend_flow_module(f::CompositeFlow, new_module::F) where F<:AbstractFlow

Prepend the chain of flow modules in `f` with `new_module`. Meaning that `new_module` 
will be applied first in the resulting flow.
"""
function prepend_flow_module(f::CompositeFlow, new_module::F) where F<:AbstractFlow
    return CompositeFlow([new_module, f.flow.fs...])
end
export prepend_flow_module

"""
    append_flow_module(f::CompositeFlow, new_module::F) where F<:AbstractFlow

Append `new_module` to the the chain of flow modules in `f`. Meaning that `new_module` 
will be applied last in the resulting flow.
"""
function append_flow_module(f::CompositeFlow, new_module::F) where F<:AbstractFlow
    return CompositeFlow([f.flow.fs..., new_module])
end
export append_flow_module

"""
    AbstractFlowModule <: AbstractFlow

A flow module is a normalizing flow that transforms each of the input's components. 
It differs from a flow block, in that flow blocks may only partially transform an input.
A flow module may consist of a scaling and shifting operation of the input samples, or be a composition of 
several flow blocks of a specific type (see `AbstractFlowBlock`). 
"""
abstract type AbstractFlowModule <: AbstractFlow
end

struct FlowModule <: AbstractFlowModule
    flow::Function
    trainable::Bool
end

export FlowModule
@functor FlowModule

function InverseFunctions.inverse(f::FlowModule)
    return FlowModule(InverseFunctions.inverse(f.flow), f.trainable)
end

"""
    AbstractFlowBlock <: AbstractFlowModule

A flow block is a normalizing flow that may only 
transform a fraction of the components of the input samples. To transform all components of the input, 
several flow blocks must be composed to a flow module (see `AbstractFlowModule`).
"""
abstract type AbstractFlowBlock <: AbstractFlowModule
end

"""
    build_flow(n_dims::Integer, modules::Vector, compute_unit::AbstractComputeUnit = CPUnit())

Construct a `CompositeFlow` to transfrom samples from a `n_dims` -dimensional target distribution, 
with the component modules in `modules`. The flow is initialized to target objects stored on `compute_unit` (defaults to CPU)
The first entry in `modules` is the function that is applied first to inputs of the resulting `CompositeFlow`.

The entries in `modules` may be actual functions or the names of the objects desired. 
"""
function build_flow(n_dims::Integer, modules::Vector, compute_unit::AbstractComputeUnit = CPUnit())
    @argcheck !any((broadcast(x -> x <: AffineMaps.AbstractAffineMap))) throw(DomainError(modules, "One or more of the specified modules are uninitailized and depend on the target input. Please use `build_flow(target_samples, modules)` to initialize modules depending on the target_samples."))
    flow_modules = Function[flow_module isa Function ? flow_module : flow_module(n_dims, compute_unit = compute_unit) for flow_module in modules]

    isa_flow = broadcast(flow_module -> flow_module isa AbstractFlow, flow_modules)
    broadcast!(flow_module -> FlowModule(flow_module, _is_trainable(flow_module)), flow_modules[.! isa_flow])

    return CompositeFlow(flow_modules)
end

function build_flow(target_samples::AbstractArray, modules::Vector = [InvMulAdd, RQSplineCouplingModule], compute_unit::AbstractComputeUnit = CPUnit())
    # n_dims = target_samples isa Matrix ? size(target_samples, 1) : (target_samples isa ArrayOfSimilarArrays ? size(target_samples.data, 1) : throw(DomainError(target_samples, "Please input the target samples either as a `Matrix` or an `ArrayOfSimilarArrays`")))

    flat_samples = flatview(target_samples)
    n_dims = size(flat_samples, 1)

    trainable =_is_trainable.(modules) 
    flow_modules = Vector{Function}(undef, length(modules))

    if any(trainable)
        flow_modules[trainable] = Function[flow_module isa Function ? flow_module : flow_module(n_dims, compute_unit = compute_unit) for flow_module in modules[trainable]]
    end

    if !trainable[1] 
        stds = vec(std(flat_samples, dims = 2))
        means = vec(mean(flat_samples, dims = 2))

        flow_modules[1] = modules[1] isa Function ? typeof(modules[1])(Diagonal(stds), means) : modules[1](Diagonal(stds), means)
    end

    for (i, flow_module) in enumerate(modules[2:end])
        if !trainable[i + 1] 
            y_intermediate = fchain(flow_modules[1:i])(flat_samples)
            stds = vec(std(y_intermediate, dims = 2))
            means = vec(mean(y_intermediate, dims = 2))

            flow_modules[i + 1] = flow_module isa Function ? typeof(flow_module)(Diagonal(stds), means) : flow_module(Diagonal(stds), means)
        end
    end

    isa_flow = broadcast(flow_module -> flow_module isa AbstractFlow, flow_modules)
    #broadcast!(flow_module -> FlowModule(flow_module), flow_modules[.! isa_flow])
    
    flow_modules[.! isa_flow] = broadcast(flow_module -> FlowModule(flow_module, _is_trainable(flow_module)), flow_modules[.! isa_flow])

    CompositeFlow(flow_modules)
end
export build_flow


function _is_trainable(flow)

    if flow isa FlowModule && !flow.trainable
        return false
    end

    if flow isa CompositeFlow
        return any(_is_trainable.(flow.flow.fs))
    end

    if typeof(flow) <: Function
        return flow isa AbstractFlowModule
    end
    
    return flow <: AbstractFlowModule
end
