# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

struct RQSplineCouplingBlock <: Function
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
