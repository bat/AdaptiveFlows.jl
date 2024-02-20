# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2
std_normal_logpdf(x::AbstractArray) = vec(sum(std_normal_logpdf.(flatview(x)), dims = 1))

function negll_flow_loss(flow::F, x::AbstractMatrix{<:Real}, logd_orig::AbstractVector, logpdf::Function) where F<:AbstractFlow
    nsamples = size(x, 2) 
    flow_corr = fchain(flow,logpdf.f)
    y, ladj = with_logabsdet_jacobian(flow_corr, x)
    ll = (sum(logpdf.logdensity(y)) + sum(ladj)) / nsamples
    return -ll
end

function negll_flow(flow::F, x::AbstractMatrix{<:Real}, logd_orig::AbstractVector, logpdf::Tuple{Function, Function}) where F<:AbstractFlow
    negll, back = Zygote.pullback(negll_flow_loss, flow, x, logd_orig, logpdf[2])
    d_flow = back(one(eltype(x)))[1]
    return negll, d_flow
end
export negll_flow

function KLDiv_flow_loss(flow::F, x::AbstractMatrix{<:Real}, logd_orig::AbstractVector, logpdf::Function) where F<:AbstractFlow
    nsamples = size(x, 2) 
    flow_corr = fchain(flow,logpdf.f)
    #logpdf_y = logpdfs[2].logdensity
    y, ladj = with_logabsdet_jacobian(flow_corr, x)
    KLDiv = sum(exp.(logd_orig - vec(ladj)) .* (logd_orig - vec(ladj) - logpdf(y))) / nsamples
    return KLDiv
end

function KLDiv_flow(flow::F, x::AbstractMatrix{<:Real}, logd_orig::AbstractVector, logpdf::Tuple{Function, Function}) where F<:AbstractFlow
    KLDiv, back = Zygote.pullback(KLDiv_flow_loss, flow, x, logd_orig, logpdf[2])
    d_flow = back(one(eltype(x)))[1]
    return KLDiv, d_flow
end
export KLDiv_flow


function optimize_flow(samples::Union{Matrix, Tuple{Matrix, Matrix}}, 
    initial_flow::F where F<:AbstractFlow, 
    optimizer;
    sequential::Bool = true,
    loss::Function = negll_flow_grad,
    logpdf::Union{Function, Tuple{Function, Function}} = std_normal_logpdf,
    nbatches::Integer = 10, 
    nepochs::Integer = 100,
    loss_history = Vector{Float64}(),
    shuffle_samples::Bool = false
    )
    optimize_flow(nestedview(samples), 
        initial_flow, 
        optimizer;
        sequential = sequential,
        loss = loss,
        logpdf = logpdf,
        nbatches = nbatches, 
        nepochs = nepochs,
        loss_history = loss_history,
        shuffle_samples = shuffle_samples
        )
end

function optimize_flow(samples::Union{AbstractArray, Tuple{AbstractArray, AbstractArray}}, 
    initial_flow::F where F<:AbstractFlow, 
    optimizer;
    sequential::Bool = true,
    loss::Function = negll_flow_grad,
    logpdf::Union{Function, Tuple{Function, Function}},
    nbatches::Integer = 10, 
    nepochs::Integer = 100,
    loss_history = Vector{Float64}(),
    shuffle_samples::Bool = false
    )
    if !_is_trainable(initial_flow)
        return (result = initial_flow, optimizer_state = nothing, loss_history = nothing)
    end 
    
    n_dims = _get_n_dims(samples) 
    logd_orig = samples isa Tuple ? logpdf[1](samples[1]) : logpdf[1](samples)
    pushfwd_logpdf = logpdf[2] == std_normal_logpdf ? (PushForwardLogDensity(first(initial_flow.flow.fs), logpdf[1]), PushForwardLogDensity(FlowModule(InvMulAdd(I(n_dims), zeros(n_dims)), false), logpdf[2])) : (PushForwardLogDensity(first(initial_flow.flow.fs), logpdf[1]), PushForwardLogDensity(last(initial_flow.flow.fs), logpdf[2]))

    if sequential 
        flow, state, loss_hist = _train_flow_sequentially(samples, initial_flow, optimizer, nepochs, nbatches, loss, pushfwd_logpdf, logd_orig, shuffle_samples)
    else 
        flow, state, loss_hist = _train_flow(samples, initial_flow, optimizer, nepochs, nbatches, loss, pushfwd_logpd, logd_orig, shuffle_samples)
    end

    return (result = flow, optimizer_state = state, loss_hist = vcat(loss_history, loss_hist))
end
export optimize_flow

function _train_flow_sequentially(samples::Union{AbstractArray, Tuple{AbstractArray, AbstractArray}}, 
                                  initial_flow::AbstractFlow, 
                                  optimizer, 
                                  nepochs::Integer, 
                                  nbatches::Integer, 
                                  loss::Function, 
                                  pushfwd_logpdf::Union{Function, 
                                  Tuple{Function, Function}}, 
                                  logd_orig::AbstractVector, 
                                  shuffle_samples::Bool)
    
    if !_is_trainable(initial_flow)
        return initial_flow, nothing, nothing
    end

    if initial_flow isa CompositeFlow || initial_flow isa AbstractFlowModule
        trained_components = Vector{AbstractFlow}()
        component_optstates = Vector{Any}()
        component_loss_hists = Vector{Any}()
        intermediate_samples = samples
        logd_orig_intermediate = logd_orig

        for flow_component in initial_flow.flow.fs
            trained_flow_component, component_opt_state, component_loss_hist = _train_flow_sequentially(intermediate_samples, 
                                                                                                        flow_component, 
                                                                                                        optimizer, 
                                                                                                        nepochs, 
                                                                                                        nbatches, 
                                                                                                        loss, 
                                                                                                        pushfwd_logpdf, 
                                                                                                        logd_orig_intermediate, 
                                                                                                        shuffle_samples)
            push!(trained_components, trained_flow_component)
            push!(component_optstates, component_opt_state)
            push!(component_loss_hists, component_loss_hist)

            if samples isa Tuple
                x_int, ladj = with_logabsdet_jacobian(trained_flow_component, intermediate_samples[1])
                intermediate_samples = (x_int, trained_flow_component(intermediate_samples[2]))
                # fix AffineMaps to return row matrix ladj
                ladj = ladj isa Real ? fill(ladj, length(logd_orig_intermediate)) : vec(ladj)
                logd_orig_intermediate -= ladj
            else
                intermediate_samples, ladj = with_logabsdet_jacobian(trained_flow_component, intermediate_samples)
                ladj = ladj isa Real ? fill(ladj, length(logd_orig_intermediate)) : vec(ladj)
                logd_orig_intermediate -= ladj
            end            
        end
        return typeof(initial_flow)(trained_components), component_optstates, component_loss_hists
    end
    _train_flow(samples, initial_flow, optimizer, nepochs, nbatches, loss, pushfwd_logpdf, logd_orig, shuffle_samples)
end


function _train_flow(samples::Union{AbstractArray, Tuple{AbstractArray, AbstractArray}}, 
                     initial_flow::AbstractFlow, 
                     optimizer, 
                     nepochs::Integer, 
                     nbatches::Integer, 
                     loss::Function, 
                     pushfwd_logpdf::Union{Function, Tuple{Function, Function}}, 
                     logd_orig::AbstractVector,
                     shuffle_samples::Bool)

    if !_is_trainable(initial_flow)
        return initial_flow, nothing, nothing
    end
    n_samples = samples isa Tuple ? length(samples[1]) : length(samples)
    batchsize = round(Int, n_samples / nbatches)
    batches = samples isa Tuple ? collect.(Iterators.partition.(samples, batchsize)) : collect(Iterators.partition(samples, batchsize))
    logd_orig_batches = collect(Iterators.partition(logd_orig, batchsize))
    flow = deepcopy(initial_flow)
    state = Optimisers.setup(optimizer, deepcopy(initial_flow))
    loss_hist = Vector{Float64}()
    for i in 1:nepochs
        for j in 1:nbatches
            training_samples = batches isa Tuple ? (Matrix(flatview(batches[1][j])), Matrix(flatview(batches[2][j]))) : Matrix(flatview(batches[j]))
            loss_val, d_flow = loss(flow, training_samples, logd_orig_batches[j], pushfwd_logpdf)
            state, flow = Optimisers.update(state, flow, d_flow)
            push!(loss_hist, loss_val)
        end
        if shuffle_samples
            batches = collect(Iterators.partition(shuffle(samples), batchsize))
            #TODO also shuffle logd_orig_batches
        end
    end
    return flow, state, loss_hist
end

function _get_n_dims(samples::Union{AbstractArray, Tuple{AbstractArray, AbstractArray}})
    if samples isa Tuple
        x = samples[1]
        n_dims = x isa Matrix ? size(x, 1) : (x isa ArraysOfArrays.ArrayOfSimilarArrays ? size(x.data, 1) : throw(DomainError(x, "Please input the target samples either as a `Matrix` or an `ArrayOfSimilarArrays`")))
    else
        n_dims = samples isa Matrix ? size(samples, 1) : (samples isa ArraysOfArrays.ArrayOfSimilarArrays ? size(samples.data, 1) : throw(DomainError(samples, "Please input the target samples either as a `Matrix` or an `ArrayOfSimilarArrays`")))
    end
    return n_dims 
end

struct PushForwardLogDensity{F<:Function, D<:Function} <: Function
    f::F
    logdensity::D
end
@functor PushForwardLogDensity

function (f::PushForwardLogDensity)(x)
    y, ladj = with_logabsdet_jacobian(f.f, x)
    if ladj isa Real 
        return f.logdensity(y) .+ ladj
    end
    return f.logdensity(y) + vec(ladj)
end


## Experimental
function composite_loss(flow::F, samples::Tuple{AbstractMatrix{<:Real}, AbstractMatrix{<:Real}}, logd_orig::AbstractVector, logpdfs::Tuple{Function, Function}, weights::AbstractVector{<:Real}) where F<:AbstractFlow
    x, y = samples
    logpdf_x, logpdf_y = logpdfs[1].logdensity, logpdfs[2].logdensity
    nsamples = size(x, 2) 

    x_flow, ladj_x = with_logabsdet_jacobian(fchain(inverse(flow), inverse(logpdfs[1].f)), y)
    y_flow, ladj_y = with_logabsdet_jacobian(fchain(flow, logpdfs[2].f), x)
    
    D_RKL_x = weights[1] > 0 ? sum(exp.(logpdf_y(y_flow) + vec(ladj_y)) .* (logpdf_y(y_flow) + vec(ladj_y) - logpdf_x(x))) : 0
    D_RKL_y = weights[2] > 0 ? sum(exp.(logpdf_x(x_flow) + vec(ladj_x)) .* (logpdf_x(x_flow) + vec(ladj_x) - logpdf_y(y))) : 0
    
    ll_x = weights[3] > 0 ? (sum(logpdf_x(x_flow)) + sum(ladj_x)) : 0
    ll_y = weights[4] > 0 ? (sum(logpdf_y(y_flow)) + sum(ladj_y)) : 0

    # FM_x = sum(exp.(logpdf_x(x)) .*  (∇ₓlogpdf_x(x) - ∇ₓlogpdf_x_flow(x))^2)
    # FM_y = sum(exp.(logpdf_y(y)) .*  (∇ₓlogpdf_y(y) - ∇ₓlogpdf_y_flow(y))^2)

    loss =  (weights[1] * D_RKL_x + weights[2] * D_RKL_y - weights[3] * ll_x - weights[4] * ll_y) / nsamples

    return loss
end

function composite_loss_flow_grad(flow::F, samples::Tuple{AbstractMatrix{<:Real}, AbstractMatrix{<:Real}}, logd_orig::AbstractVector, logpdfs::Tuple{Function, Function}, weights::AbstractVector{<:Real} = [0, 0, 1, 0]) where F<:AbstractFlow
    loss, back = Zygote.pullback(composite_loss, flow, samples, logd_orig::AbstractVector, logpdfs, weights)
    d_flow = back(one(eltype(samples[1])))[1]
    return loss, d_flow
end
