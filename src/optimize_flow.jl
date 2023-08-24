# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2

"""
    mvnormal_negll_flow(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow

Calculate the negative log-likelihood (under a multivariate standard normal distribution) of the result 
of applying `flow` to `x`.
"""
function mvnormal_negll_flow(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow
    nsamples = size(x, 2) 
    
    y, ladj = with_logabsdet_jacobian(flow, x)
    ll = (sum(std_normal_logpdf.(y)) + sum(ladj)) / nsamples

    return -ll
end

function mvnormal_negll_flow(flow::B, X::AbstractMatrix{<:Real}) where B<:AbstractFlowBlock
    nsamples = size(X, 2) 
    
    Y, ladj = with_logabsdet_jacobian(flow, X)
    ll = (sum(std_normal_logpdf.(Y[flow.mask,:])) + sum(ladj)) / nsamples

    return -ll
end
export mvnormal_negll_flow

"""
    mvnormal_negll_flow_grad(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow

Calculate the negative log-likelihood (under a multivariate standard normal distribution) of the result 
of applying `flow` to `x` and the gradient of this value.
"""
function mvnormal_negll_flow_grad(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow
    negll, back = Zygote.pullback(mvnormal_negll_flow, flow, x)
    d_flow = back(one(eltype(x)))[1]
    return negll, d_flow
end
export mvnormal_negll_flow_grad

"""
    optimize_flow(smpls::VectorOfSimilarVectors{<:Real}, 
                  initial_flow::F where F<:AbstractFlow, 
                  optimizer;
                  nbatches::Integer = 100, 
                  nepochs::Integer = 100,
                  optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                  negll_history = Vector{Float64}(),
                  shuffle_samples::Bool = false
        )

Use `optimizer` to optimize the normalizing flow `initial_flow` to optimally transform `smpls` to follow 
a multivariate standard normal distribution. Use `nbatches` and `nepochs` respectively to specify the 
number of batches and epochs to use during training. 
If desired, set `shuffle_samples` to `true` to have the samples be shuffled in between epochs. This may 
improve the training, but increase training time.

Returns a tuple `(optimized_flow, final_optimizer_state, negll_history)` where `negll_history` is a vector 
containing the values of the loss function during training.
"""
function optimize_flow(smpls::VectorOfSimilarVectors{<:Real}, 
                       initial_flow::F where F<:AbstractFlow, 
                       optimizer;
                       nbatches::Integer = 100, 
                       nepochs::Integer = 100,
                       optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                       negll_history = Vector{Float64}(),
                       shuffle_samples::Bool = false
    )
    batchsize = round(Int, length(smpls) / nbatches)
    batches = collect(Iterators.partition(smpls, batchsize))
    flow = deepcopy(initial_flow)
    state = deepcopy(optstate)
    negll_hist = Vector{Float64}()
    for i in 1:nepochs
        for batch in batches
            negll, d_flow = mvnormal_negll_flow_grad(flow, flatview(batch))
            state, flow = Optimisers.update(state, flow, d_flow)
            push!(negll_hist, negll)
        end
        if shuffle_samples
            batches = collect(Iterators.partition(shuffle(smpls), batchsize))
        end
    end
    (result = flow, optimizer_state = state, negll_history = vcat(negll_history, negll_hist))
end
export optimize_flow


function optimize_flow_sequentially(smpls::VectorOfSimilarVectors{<:Real}, 
                                    initial_flow::CompositeFlow, 
                                    optimizer;
                                    nbatches::Integer = 100, 
                                    nepochs::Integer = 100,
                                    shuffle_samples::Bool = false
    )

    optimized_modules = Vector{AbstractFlow}(undef, length(initial_flow.flow.fs))
    module_optimizer_states = Vector{NamedTuple}(undef, length(initial_flow.flow.fs))
    module_negll_hists = Vector{Vector}(undef, length(initial_flow.flow.fs))

    for (i,flow_module) in enumerate(initial_flow.flow.fs)
        opt_module, opt_state, negll_hist = optimize_flow_sequentially(smpls, flow_module, optimizer; nbatches, nepochs, shuffle_samples)
        optimized_modules[i] = opt_module
        module_optimizer_states[i] = opt_state
        module_negll_hists[i] = negll_hist
    end

    (CompositeFlow(optimized_modules), module_optimizer_states,  module_negll_hists)
end

function optimize_flow_sequentially(smpls::VectorOfSimilarVectors{<:Real}, 
                                    initial_flow::M where M<:AbstractFlowModule, 
                                    optimizer;
                                    nbatches::Integer = 100, 
                                    nepochs::Integer = 100,
                                    shuffle_samples::Bool = false
    )

    optimized_blocks = Vector{AbstractFlow}(undef, length(initial_flow.flow_module.fs))
    block_optimizer_states = Vector{NamedTuple}(undef, length(initial_flow.flow_module.fs))
    block_negll_hists = Vector{Vector}(undef, length(initial_flow.flow_module.fs))

    for (i,block) in enumerate(initial_flow.flow_module.fs)
        opt_flow, opt_state, negll_hist = optimize_flow(smpls, block, optimizer; nbatches, nepochs, shuffle_samples = shuffle_samples)
        optimized_blocks[i] = opt_flow
        block_optimizer_states[i] = opt_state
        block_negll_hists[i] = negll_hist
    end

    (typeof(initial_flow)(optimized_blocks), block_optimizer_states,  block_negll_hists)
end
export optimize_flow_sequentially
