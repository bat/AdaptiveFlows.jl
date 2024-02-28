# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

"""
    RQSplineCouplingModule <: AbstractFlowModule

A flow module using rational quadratic spline functions for the 
transformation of input components, and a coupling approach to introducing correlation between the 
dimensions of the flow's output.
"""
struct RQSplineCouplingModule <: AbstractFlowModule
    flow::Function
end

export RQSplineCouplingModule
@functor RQSplineCouplingModule

function RQSplineCouplingModule(blocks::Vector{F}) where F <: Function
    return RQSplineCouplingModule(fchain(blocks))
end    

"""
    RQSplineCouplingModule(n_dims::Integer, 
                           block_target_elements::Vector, 
                           K::Union{Integer, Vector{Integer}} = 10,
                           compute_unit::AbstractComputeUnit = CPUnit()
        )

Construct an instance of `RQSplineCouplingModule` for a `n_dims` -dimensíonal input. Use `block_target_elements` 
to specify which block in the module transforms which components of the input. Use `K` to specify the desired 
number of spline segments used for the rational quadratic spline functions (defaults to 10). If desired, use 
`compute_unit` to specify the flow to be initiated on a different compute device, using the API of 
"HeterogeneousComputing.jl"(https://oschulz.github.io/HeterogeneousComputing.jl/stable/).

Note: This constructor does not ensure each element of the input is transformed by a block. If desired, this 
must be ensured in `block_target_elements`.

Alternative call signature:

    RQSplineCouplingModule(n_dims::Integer, 
                           block_target_elements::Integer = 1, 
                           K::Union{Integer, Vector{Integer}} = 10,
                           compute_unit::AbstractComputeUnit = CPUnit()

    )

In this method, one may only input the number of dimensions `n_dims` of the target distribution. The default rational 
quadratic coupling flow module that is constructed this way, consists of `n_dims` blocks of `RQSplineCouplingBlock`s, 
each of which transforms one component of the input and uses 10 spline segments for its spline functions. 
"""
function RQSplineCouplingModule(n_dims::Integer, 
                                block_target_elements::Vector; 
                                K::Union{Integer, Vector{Integer}} = 10,
                                compute_unit::AbstractComputeUnit = CPUnit()
    )
    @argcheck K isa Integer || length(K) == length(block_target_elements) throw(DomainError(K, "please specify the same number of values for K as there are blocks"))

    n_blocks = length(block_target_elements)
    blocks = Vector{RQSplineCouplingBlock}(undef, n_blocks)
    n_out_neural_net = K isa Vector ? 3 .* K .- 1 : 3 .* fill(K, n_blocks) .- 1

    for i in 1:n_blocks
        transformation_mask = fill(false, n_dims)
        block_target_elements[i] isa Integer ? transformation_mask[block_target_elements[i]] = true : transformation_mask[block_target_elements[i]] .= true
        neural_net = get_neural_net(n_dims - sum(transformation_mask), n_out_neural_net[i])
        blocks[i] = RQSplineCouplingBlock(transformation_mask, neural_net, compute_unit)
    end
    return RQSplineCouplingModule(fchain(blocks))
end

function RQSplineCouplingModule(n_dims::Integer, 
                                block_target_elements::Integer = 1;
                                K::Union{Integer, Vector{Integer}} = 10,
                                compute_unit::AbstractComputeUnit = CPUnit()

    )

    n_blocks = ceil(Integer, n_dims / block_target_elements)
    vectorized_bte = [UnitRange(i + 1, i + block_target_elements) for i in 0:block_target_elements:((n_blocks - 2) * block_target_elements) ]
    push!(vectorized_bte, UnitRange((n_blocks - 1) * block_target_elements + 1, n_dims))

    RQSplineCouplingModule(n_dims, vectorized_bte, K = K, compute_unit = compute_unit)
end

function InverseFunctions.inverse(f::RQSplineCouplingModule)
    return RQSplineCouplingModule(InverseFunctions.inverse(f.flow).fs)
end

abstract type AbstractRQSplineCouplingBlock <: AbstractFlowBlock
end

"""
    RQSplineCouplingBlock <: AbstractFlowBlock

An object holding the neural net and the input mask to transform samples using using rational quadratic 
spline functions for the transformation of the input components of a normalizing flow, and a coupling approach to introducing 
correlation between the dimensions of the flow's output.
"""
struct RQSplineCouplingBlock <: AbstractRQSplineCouplingBlock
    mask::Vector{Bool}
    nn::Chain
    nn_parameters::NamedTuple
    nn_state::NamedTuple
end
    
export RQSplineCouplingBlock
@functor RQSplineCouplingBlock

"""
    RQSplineCouplingBlock(mask::Vector{Bool}, nn::Chain, compute_unit::AbstractComputeUnit=CPUnit())

Construct and instance of `RQSplineCouplingBlock`, while initializing the parameters and the state of `nn` on the 
compute device specified in `compute_unit`. (Defaults to CPU)
"""
function RQSplineCouplingBlock(mask::Vector{Bool}, nn::Chain, compute_unit::AbstractComputeUnit=CPUnit())
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    lux_compute_unit = compute_unit isa CPUnit ? cpu_device() : gpu_device()

    nn_parameters, nn_state = Lux.setup(rng, nn) .|> lux_compute_unit

    return RQSplineCouplingBlock(mask, nn, nn_parameters, nn_state)
end

function InverseFunctions.inverse(f::RQSplineCouplingBlock)
    return InverseRQSplineCouplingBlock(f.mask, f.nn, f.nn_parameters, f.nn_state)
end


"""
    InverseRQSplineCouplingBlock <: AbstractFlowBlock

An object holding the neural net and the input mask to transform samples using *inverse* rational quadratic spline 
functions for the transformation of the input components of a normalizing flow with a coupling approach to introducing 
correlation between the dimensions of the flow's output.
"""
struct InverseRQSplineCouplingBlock <: AbstractRQSplineCouplingBlock
    mask::Vector{Bool}
    nn::Chain
    nn_parameters::NamedTuple
    nn_state::NamedTuple
end

export InverseRQSplineCouplingBlock
@functor InverseRQSplineCouplingBlock

"""
    InverseRQSplineCouplingBlock(mask::Vector{Bool}, nn::Chain, compute_unit::AbstractComputeUnit=CPUnit())

Construct and instance of `InverseRQSplineCouplingBlock`, while initializing the parameters and the state of `nn` on the 
compute device specified in `compute_unit`. (Defaults to CPU)
"""
function InverseRQSplineCouplingBlock(mask::Vector{Bool}, nn::Chain, compute_unit::AbstractComputeUnit=CPUnit())
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    lux_compute_unit = compute_unit isa CPUnit ? cpu_device() : gpu_device()

    nn_parameters, nn_state = Lux.setup(rng, nn) .|> lux_compute_unit

    return InverseRQSplineCouplingBlock(mask, nn, nn_parameters, nn_state)
end

function InverseFunctions.inverse(f::InverseRQSplineCouplingBlock)
    return RQSplineCouplingBlock(f.mask, f.nn, f.nn_parameters, f.nn_state)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::AbstractRQSplineCouplingBlock,
    x::Matrix
)
    apply_rqs_coupling_flow(f, x)
end

function ChangesOfVariables.with_logabsdet_jacobian(
    f::AbstractRQSplineCouplingBlock,
    x::Vector
)
    y, ladj = apply_rqs_coupling_flow(f, reshape(x, :, 1))
    return vec(y), ladj[1]
end

(f::AbstractRQSplineCouplingBlock)(x::Matrix) = apply_rqs_coupling_flow(f, x)[1]
(f::AbstractRQSplineCouplingBlock)(x::Vector) = vec(apply_rqs_coupling_flow(f, reshape(x , :, 1))[1])
(f::AbstractRQSplineCouplingBlock)(vs::AbstractValueShape) = vs


"""
    apply_rqs_coupling_flow(flow::Union{RQSplineCouplingBlock, InverseRQSplineCouplingBlock}, x::Any)

Apply the flow block `flow` to the input `x`, and compute the logarithm of the absolute value of the jacobian of this transformation. 
Returns a tuple with the transformed output in the first component and a row matrix of the corresponding log values of the abs of 
the jacobians in the second component.
"""
function apply_rqs_coupling_flow(flow::Union{RQSplineCouplingBlock, InverseRQSplineCouplingBlock}, x::AbstractArray)

    rq_spline = flow isa RQSplineCouplingBlock ? RQSpline : InvRQSpline
    n_dims_to_transform = sum(flow.mask)

    input_mask = .~flow.mask 
    y, ladj = with_logabsdet_jacobian(rq_spline(get_params(flow.nn(x[input_mask,:], flow.nn_parameters, flow.nn_state)[1], n_dims_to_transform)...), x[flow.mask,:])   

    return MonotonicSplines._sort_dimensions(y, x, flow.mask), ladj
end

export apply_rqs_coupling_flow
