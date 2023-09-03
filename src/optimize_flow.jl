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
                  loss_history = Vector{Float64}(),
                  shuffle_samples::Bool = false
        )

Use `optimizer` to optimize the normalizing flow `initial_flow` to optimally transform `smpls` to follow 
a multivariate standard normal distribution. Use `nbatches` and `nepochs` respectively to specify the 
number of batches and epochs to use during training. 
If desired, set `shuffle_samples` to `true` to have the samples be shuffled in between epochs. This may 
improve the training, but increase training time.

Returns a tuple `(optimized_flow, final_optimizer_state, loss_history)` where `loss_history` is a vector 
containing the values of the loss function during training.
"""
function optimize_flow(smpls::VectorOfSimilarVectors{<:Real}, 
                       initial_flow::F where F<:AbstractFlow, 
                       optimizer;
                       nbatches::Integer = 100, 
                       nepochs::Integer = 100,
                       optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                       loss_history = Vector{Float64}(),
                       shuffle_samples::Bool = false
    )
    if initial_flow isa ScaleShiftModule
        return (result = initial_flow, optimizer_state = nothing, loss_history = nothing)
    end

    batchsize = round(Int, length(smpls) / nbatches)
    batches = collect(Iterators.partition(smpls, batchsize))
    flow = deepcopy(initial_flow)
    state = deepcopy(optstate)
    loss_hist = Vector{Float64}()
    for i in 1:nepochs
        for batch in batches
            loss, d_flow = mvnormal_negll_flow_grad(flow, flatview(batch))
            state, flow = Optimisers.update(state, flow, d_flow)
            push!(loss_hist, loss)
        end
        if shuffle_samples
            batches = collect(Iterators.partition(shuffle(smpls), batchsize))
        end
    end
    (result = flow, optimizer_state = state, loss_history = vcat(loss_history, loss_hist))
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
    module_optimizer_states = Vector(undef, length(initial_flow.flow.fs))
    module_loss_hists = Vector{Vector}(undef, length(initial_flow.flow.fs))

    for (i,flow_module) in enumerate(initial_flow.flow.fs)
        opt_module, opt_state, loss_hist = optimize_flow_sequentially(smpls, flow_module, optimizer; nbatches, nepochs, shuffle_samples)
        optimized_modules[i] = opt_module
        module_optimizer_states[i] = opt_state
        module_loss_hists[i] = loss_hist
    end

    (result = CompositeFlow(optimized_modules), optimizer_states = module_optimizer_states, loss_histories = module_loss_hists)
end

function optimize_flow_sequentially(smpls::VectorOfSimilarVectors{<:Real}, 
                                    initial_flow::M where M<:AbstractFlowModule, 
                                    optimizer;
                                    nbatches::Integer = 100, 
                                    nepochs::Integer = 100,
                                    shuffle_samples::Bool = false
    )
    @argcheck !(initial_flow isa AbstractFlowBlock) throw DomainError("The input flow is an individual flow block, please use `optimize_flow()`[@ref] to optimize flow blocks.")
    
    if initial_flow isa ScaleShiftModule
        return (result = initial_flow, optimizer_states = nothing, loss_hists = nothing)
    end

    optimized_blocks = Vector{AbstractFlow}(undef, length(initial_flow.flow_module.fs))
    block_optimizer_states = Vector{NamedTuple}(undef, length(initial_flow.flow_module.fs))
    block_loss_hists = Vector{Vector}(undef, length(initial_flow.flow_module.fs))

    for (i,block) in enumerate(initial_flow.flow_module.fs)
        opt_flow, opt_state, loss_hist = optimize_flow(smpls, block, optimizer; nbatches, nepochs, shuffle_samples = shuffle_samples)
        optimized_blocks[i] = opt_flow
        block_optimizer_states[i] = opt_state
        block_loss_hists[i] = loss_hist
    end

    (result = typeof(initial_flow)(optimized_blocks), optimizer_states = block_optimizer_states,  loss_hists = block_loss_hists)
end
export optimize_flow_sequentially
