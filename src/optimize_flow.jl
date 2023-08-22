# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2

function mvnormal_negll_flow(flow::Function, X::AbstractMatrix{<:Real})
    nsamples = size(X, 2) 
    
    Y, ladj = with_logabsdet_jacobian(flow, X)
    ll = (sum(std_normal_logpdf.(Y[flow.mask,:])) + sum(ladj)) / nsamples

    return -ll
end

function mvnormal_negll_flow_grad(flow, X::AbstractMatrix{<:Real})
    negll, back = Zygote.pullback(mvnormal_negll_flow, flow, X)
    d_flow = back(one(eltype(X)))[1]
    return negll, d_flow
end


function optimize_flow(smpls::VectorOfSimilarVectors{<:Real}, 
                       initial_flow, 
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

# temporary hack
function optimize_flow_sequentially(smpls::VectorOfSimilarVectors{<:Real}, 
                                    initial_flow::CompositeFlow, 
                                    optimizer;
                                    nbatches::Integer = 100, 
                                    nepochs::Integer = 100,
                                    optstate = Optimisers.setup(optimizer, deepcopy(initial_flow)),
                                    negll_history = Vector{Float64}(),
                                    shuffle_samples::Bool = false
    )

    optimized_blocks = Vector{Function}(undef, length(initial_flow.flow.fs))
    for block in initial_flow.flow.fs
        res = optimize_flow(smpls, block, optimizer; nbatches, nepochs, optstate, negll_history, shuffle_samples)
        optimized_blocks[i] = res.result
    end
    return CompositeFlow(optimized_blocks)
end

export optimize_flow_sequentially
