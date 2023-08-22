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


struct RQSplineCouplingModule <: AbstractFlow
    flow_module::Function
end

export RQSplineCouplingModule
@functor RQSplineCouplingModule


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
    return fchain(blocks)
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
