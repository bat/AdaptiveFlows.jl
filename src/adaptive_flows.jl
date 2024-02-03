# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

""" 
    AbstractFlow <: Function

Supertype for functions used as "normalizing flows" for transforming complex probability
distributions into simpler ones.

A normalizing flow is an invertible, differentiable function mapping a `D`-dimensional 
space onto another `D`-dimensional space. 

When applied to a matrix of samples from a probability distribution, a normalizing flow
returns a tuple: the transformed output and a row matrix with the `1,i`-th element 
holding the logarithm of the absolute value of the determinant of the Jacobian of the 
transformation, applied to the `ì`-th sample in the input. 
"""
abstract type AbstractFlow <: Function
end

export AbstractFlow

"""
    CompositeFlow <: AbstractFlow

Represents a composition of multiple normalizing flows, each transforming all components
of the input data.

# Fields:

flow<:Function: A function representing the composed normalizing flow. 
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

function ChangesOfVariables.with_logabsdet_jacobian(
    f::F,
    x::ArrayOfSimilarArrays
) where F <: AbstractFlow
    y, ladj = with_logabsdet_jacobian(f, flatview(x))
    return nestedview(y), ladj
end

(f::AbstractFlow)(x::Matrix) = f.flow(x)
(f::AbstractFlow)(x::Vector) = vec(f.flow(reshape(x, :, 1)))
(f::AbstractFlow)(x::ArrayOfSimilarArrays) = nestedview(f(flatview(x)))
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

A flow module is a normalizing flow that transforms each of a multidimensional input's 
dimensions. 
It differs from a flow block, in that flow blocks may only partially transform an input.
A flow module may consist of a scaling and shifting operation of the input samples, or be
a composition of several flow blocks of a specific type (see `AbstractFlowBlock`). 
"""
abstract type AbstractFlowModule <: AbstractFlow
end

export AbstractFlowModule

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
    AbstractFlowBlock <: AbstractFlow

A flow block is a normalizing flow that can only transform a fraction of the dimensions of
a multidimensional input. 
To transform all components of the input, several flow blocks must be composed to a flow
module (see `AbstractFlowModule`).
"""
abstract type AbstractFlowBlock <: AbstractFlow
end
# is not a subtype of AbstractFlowModule to facilitate distinction in flow optimization

export AbstractFlowBlock

"""
    build_flow(n_dims::Integer, 
               modules::Vector, 
               compute_unit::AbstractComputeUnit = CPUnit())

Compose a number of normalizing flows into a `CompositeFlow` to transform samples from a
`n_dims`-dimensional target distribution. 
The resulting flow is initialized to target objects stored on `compute_unit` 
(defaults to CPU).

# Arguments
- `n_dims::Integer`: The number of dimensions of the target distribution.
- `modules::Vector`: A vector of functions or types that will be used as the component
                     modules of the `CompositeFlow`. The modules are applied in the order
                     they appear in in `modules`.
- `compute_unit::AbstractComputeUnit`: (optional) The computing device where the target
                     objects are stored. Defaults to CPU.

# Examples
#TODO
```julia
flow = build_flow(3, [module1, module2, module3])

This will create a CompositeFlow that transforms 3-dimensional data using `module1`, 
`module2`, and `module3` in that order. 
"""
function build_flow(n_dims::Integer, modules::Vector, compute_unit::AbstractComputeUnit = CPUnit())
    @argcheck !any((broadcast(x -> x <: AffineMaps.AbstractAffineMap))) throw(DomainError(modules, "One or more of the specified modules are uninitailized and depend on the target input. Please use `build_flow(target_samples, modules)` to initialize modules depending on the target_samples."))
    flow_modules = Function[flow_module isa Function ? flow_module : flow_module(n_dims, compute_unit = compute_unit) for flow_module in modules]

    isa_flow = broadcast(flow_module -> flow_module isa AbstractFlow, flow_modules)
    broadcast!(flow_module -> FlowModule(flow_module, _is_trainable(flow_module)), flow_modules[.! isa_flow])

    return CompositeFlow(flow_modules)
end

"""
    build_flow(target_samples::AbstractArray, 
               modules::Vector = [InvMulAdd, RQSplineCouplingModule],
               compute_unit::AbstractComputeUnit = CPUnit())

Construct a `CompositeFlow` suited for transforming samples similar to `target_samples`.
The resulting flow is initialized to target objects stored on `compute_unit` 
(defaults to CPU).

# Arguments
- `target_samples::AbstractArray`: A set of samples from the probability distribution the
                                   flow is intended to target. Has to be in the shape 
                                   `n_dims x n_samples`, where `n_dims` is the number of 
                                   dimensions of the target distribution.
- `modules::Vector`: (optional) A vector of functions or types that will be used as the 
                     component modules of the `CompositeFlow`. The modules are applied 
                     in the order they appear in in `modules`.
- `compute_unit::AbstractComputeUnit`: (optional) The computing device where the target
                     objects are stored. Defaults to CPU.

# Examples
#TODO
"""
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
        return flow isa AbstractFlowModule || flow isa AbstractFlowBlock
    end
    
    return flow <: AbstractFlowModule || flow <: AbstractFlowBlock
end
