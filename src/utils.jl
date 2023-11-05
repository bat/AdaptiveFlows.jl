# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

function get_neural_net(n_in::Integer, 
                        n_out::Integer, 
                        n_hidden_layers::Integer = 1, 
                        n_in_hidden::Integer = 20
                        #compute_device::AbstractComputeDevice = CPUnit(),
                    )
                    
    layers = vcat([Dense(n_in, n_in_hidden, relu)], repeat([Dense(n_in_hidden, n_in_hidden, relu)], n_hidden_layers), [Dense(n_in_hidden, n_out)])
    return Chain(layers...)
end
