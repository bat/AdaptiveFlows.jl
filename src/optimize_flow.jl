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

"""
    mvnormal_negll_flow(flow::B, x::AbstractMatrix{<:Real}) where B<:AbstractFlowBlock

Calculate the negative log-likelihood (under a multivariate standard normal distribution) of the result 
of applying `flow` to `x`.
"""
function mvnormal_negll_flow(flow::B, x::AbstractMatrix{<:Real}) where B<:AbstractFlowBlock
    nsamples = size(x, 2) 
    
    y, ladj = with_logabsdet_jacobian(flow, x)
    ll = (sum(std_normal_logpdf.(y[flow.mask,:])) + sum(ladj)) / nsamples

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
    optimize_flow(samples::VectorOfSimilarVectors{<:Real}, 
                  initial_flow::F where F<:AbstractFlow, 
                  optimizer;
                  nbatches::Integer = 100, 
                  nepochs::Integer = 100,
                  optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                  loss_history = Vector{Float64}(),
                  shuffle_samples::Bool = false
        )

Use `optimizer` to optimize the normalizing flow `initial_flow` to optimally transform `samples` to follow 
a multivariate standard normal distribution. Use `nbatches` and `nepochs` respectively to specify the 
number of batches and epochs to use during training. 
If desired, set `shuffle_samples` to `true` to have the samples be shuffled in between epochs. This may 
improve the training, but increase training time.

Returns a `NamedTuple` `(result = optimized_flow, optimizer_state = final_optimizer_state, loss_hist = loss_history)` where `loss_history` is a vector 
containing the values of the loss function for each iteration during training.
"""
function optimize_flow(samples::AbstractArray, 
                       initial_flow::F where F<:AbstractFlow, 
                       optimizer;
                       nbatches::Integer = 10, 
                       nepochs::Integer = 100,
                       optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                       loss_history = Vector{Float64}(),
                       shuffle_samples::Bool = false
    )
    if !_is_trainable(initial_flow)
        return (result = initial_flow, optimizer_state = nothing, loss_history = nothing)
    end

    batchsize = round(Int, length(samples) / nbatches)
    batches = collect(Iterators.partition(samples, batchsize))
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
            batches = collect(Iterators.partition(shuffle(samples), batchsize))
        end
    end

    return (result = flow, optimizer_state = state, loss_hist = vcat(loss_history, loss_hist))
end
export optimize_flow

"""
    optimize_flow_sequentially(samples::AbstractArray, 
                                        initial_flow::CompositeFlow, 
                                        optimizer;
                                        nbatches::Integer = 100, 
                                        nepochs::Integer = 100,
                                        shuffle_samples::Bool = false
        )

Use `optimizer` to optimize the normalizing flow `initial_flow` to optimally transform `samples` to follow 
a multivariate standard normal distribution. 
In contrast to `optimize_flow()`, this function optimizes each component of `initial_flow` in sequence. Meaning 
that at first, the first component of `initial_flow` is optimized, then the resulting optimized component is applied 
to the input samples, which are then used to optimize the second component and so on. 
If a component of `initial_flow` is itself a composite of several component functions, these sub-component functions 
are also optimized sequentially.
Use `nbatches` and `nepochs` respectively to specify the number of batches and epochs to use during training. 
If desired, set `shuffle_samples` to `true` to have the samples be shuffled in between epochs. This may 
improve the training, but increase training time.

Returns a `NamedTuple` `(result = optimized_flow, optimizer_states = final_optimizer_states, loss_hists = loss_histories)` where `loss_hists` is a vector 
containing vectors of the values of the loss function for each iteration during training for each of the components of the input flow.
"""
function optimize_flow_sequentially(samples::AbstractArray, 
                                    initial_flow::CompositeFlow, 
                                    optimizer;
                                    nbatches::Integer = 10, 
                                    nepochs::Integer = 100,
                                    shuffle_samples::Bool = false
    )

    optimized_modules = Vector{AbstractFlow}(undef, length(initial_flow.flow.fs))
    module_optimizer_states = Vector(undef, length(initial_flow.flow.fs))
    module_loss_histories = Vector(undef, length(initial_flow.flow.fs))

    intermediate_samples = flatview(samples)

    for (i,flow_module) in enumerate(initial_flow.flow.fs)
        opt_module, opt_state, loss_hist = optimize_flow_sequentially(intermediate_samples, flow_module, optimizer; nbatches, nepochs, shuffle_samples)
        optimized_modules[i] = opt_module
        module_optimizer_states[i] = opt_state
        module_loss_histories[i] = loss_hist

        intermediate_samples = opt_module(intermediate_samples)
    end

    return (result = CompositeFlow(optimized_modules), optimizer_states = module_optimizer_states, loss_hists = module_loss_histories)
end

function optimize_flow_sequentially(samples::AbstractArray, 
                                    initial_flow::M where M<:AbstractFlowModule, 
                                    optimizer;
                                    nbatches::Integer = 10, 
                                    nepochs::Integer = 100,
                                    shuffle_samples::Bool = false
    )
    @argcheck !(initial_flow isa AbstractFlowBlock) throw DomainError("The input flow is an individual flow block, please use `optimize_flow()`[@ref] to optimize flow blocks.")
    
    if !_is_trainable(initial_flow)
        return (result = initial_flow, optimizer_states = nothing, loss_hists = nothing)
    end

    optimized_blocks = Vector{Function}(undef, length(initial_flow.flow.fs))
    block_optimizer_states = Vector{NamedTuple}(undef, length(initial_flow.flow.fs))
    block_loss_histories = Vector{Vector}(undef, length(initial_flow.flow.fs))

    intermediate_samples = samples

    for (i, block) in enumerate(initial_flow.flow.fs)
        optimized_block, optimizer_state, loss_history = optimize_flow(nestedview(intermediate_samples), block, optimizer; nbatches, nepochs, shuffle_samples = shuffle_samples)
        optimized_blocks[i] = optimized_block
        block_optimizer_states[i] = optimizer_state
        block_loss_histories[i] = loss_history

        intermediate_samples = optimized_block(intermediate_samples)
    end

    return (result = typeof(initial_flow)(optimized_blocks), optimizer_states = block_optimizer_states,  loss_hists = block_loss_histories)
end
export optimize_flow_sequentially
