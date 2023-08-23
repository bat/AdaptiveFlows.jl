# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

struct RQSplineCouplingModule <: AbstractFlowModule
    flow_module::Function
end

export RQSplineCouplingModule
@functor RQSplineCouplingModule

function RQSplineCouplingModule(blocks::Vector{F}) where F <: Function
    return RQSplineCouplingModule(fchain(blocks))
end    

"""
    RQSplineCouplingModule(n_dims::Integer, 
                           block_target_elements::Union{Vector{Vector{I}}, Vector{UnitRange{I}}} where I <: Integer, 
                           K::Union{Integer, Vector{Integer}} = 10
        )
Construct an instance of `RQSplineCouplingModule` for a `ǹ_dims` -dimensíonal input. Use `block_target_elements` 
to specify which block in the module transforms which components of the input. Use `K` to specify the desired 
number of spline segments used for the rational quadratic spline functions (defaults to 10). 

Note: This constructor does not ensure each element of the input is transformed by a block. If desired, this 
must be ensured in `block_target_elements`.
"""
function RQSplineCouplingModule(n_dims::Integer, 
                                block_target_elements::Union{Vector{Vector{I}}, Vector{UnitRange{I}}} where I <: Integer, 
                                K::Union{Integer, Vector{Integer}} = 10,
                                compute_unit::AbstractComputeUnit = CPUnit()
    )
    @argcheck K isa Integer || length(K) == length(block_target_elements) throw(DomainError(K, "please specify the same number of values for K as there are blocks"))

    n_blocks = length(block_target_elements)
    blocks = Vector{RQSplineCouplingBlock}(undef, n_blocks)
    n_out_neural_net = K isa Vector ? 3 .* K .- 1 : 3 .* fill(K, n_blocks) .- 1

    for i in 1:n_blocks
        transformation_mask = fill(false, n_dims)
        transformation_mask[block_target_elements[i]] .= true
        neural_net = get_neural_net(n_dims - sum(transformation_mask), n_out_neural_net[i])
        blocks[i] = RQSplineCouplingBlock(transformation_mask, neural_net, compute_unit)
    end
    return RQSplineCouplingModule(fchain(blocks))
end

function RQSplineCouplingModule(n_dims::Integer, 
                                block_target_elements::Integer = 1, 
                                K::Union{Integer, Vector{Integer}} = 10
    )

    n_blocks = ceil(Integer, n_dims / block_target_elements)
    vectorized_bte = [UnitRange(i + 1, i + block_target_elements) for i in range(start = 0, stop = (n_blocks - 2) * block_target_elements, step = block_target_elements)]
    push!(vectorized_bte, UnitRange((n_blocks - 1) * block_target_elements + 1, n_dims))

    RQSplineCouplingModule(n_dims, vectorized_bte, K)
end

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCouplingModule,
    x::Any
)
    with_logabsdet_jacobian(f.flow_module, x)
end

(f::RQSplineCouplingModule)(x::Any) = f.flow_module(x)
(f::RQSplineCouplingModule)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::RQSplineCouplingModule)
    return RQSplineCouplingModule(InverseFunctions.inverse(f.flow_module).fs)
end


struct RQSplineCouplingBlock <: AbstractFlowBlock
    mask::Vector{Bool}
    nn::Chain
    nn_parameters::NamedTuple
    nn_state::NamedTuple
end
    
export RQSplineCouplingBlock
@functor RQSplineCouplingBlock

function RQSplineCouplingBlock(mask::Vector{Bool}, nn::Chain, compute_unit::AbstractComputeUnit)
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    lux_compute_unit = compute_unit isa CPUnit ? cpu_device() : gpu_device()

    nn_parameters, nn_state = Lux.setup(rng, nn) .|> lux_compute_unit

    return RQSplineCouplingBlock(mask, nn, nn_parameters, nn_state)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCouplingBlock,
    x::Any
)
    apply_rqs_coupling_flow(f, x)
end

(f::RQSplineCouplingBlock)(x::Any) = apply_rqs_coupling_flow(f, x)[1]
(f::RQSplineCouplingBlock)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::RQSplineCouplingBlock)
    return InverseRQSplineCouplingBlock(f.nn, f.mask)
end


struct InverseRQSplineCouplingBlock <: Function
    mask::Vector{Bool}
    nn::Chain
end

export InverseRQSplineCouplingBlock
@functor InverseRQSplineCouplingBlock

function ChangesOfVariables.with_logabsdet_jacobian(
    f::InverseRQSplineCouplingBlock,
    x::Any
)
    return apply_rqs_coupling_flow(f, x)
end

(f::InverseRQSplineCouplingBlock)(x::Any) = apply_rqs_coupling_flow(f, x)[1]
(f::InverseRQSplineCouplingBlock)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::InverseRQSplineCouplingBlock)
    return RQSplineCouplingBlock(f.nn, f.mask)
end


function apply_rqs_coupling_flow(flow::Union{RQSplineCouplingBlock, InverseRQSplineCouplingBlock}, x::Any) # make x typestable

    rq_spline = flow isa RQSplineCouplingBlock ? RQSpline : InvRQSpline
    n_dims_to_transform = sum(flow.mask)

    input_mask = .~flow.mask 
    y, ladj = with_logabsdet_jacobian(rq_spline(get_params(flow.nn(x[input_mask,:], flow.nn_parameters, flow.nn_state)[1], n_dims_to_transform)...), x[flow.mask,:])   

    return MonotonicSplines._sort_dimensions(y, x, flow.mask), ladj
end

export apply_rqs_coupling_flow
