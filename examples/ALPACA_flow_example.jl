using Pkg
Pkg.add("https://github.com/bat/AdaptiveFlows.jl.git")
Pkg.add("BAT")
Pkg.add("Optimisers")
Pkg.add("Plots")

using AdaptiveFlows
using BAT
using Optimisers
using Plots

# create target distribution 
μ = BAT.example_posterior()

# sample target distribution 
samples_target_weighted = BAT.bat_sample(μ, MCMCSampling()).result

# IMPORTANT: use samples with unity weight, otherwise the flow may learn an incorrect distribution
samples_target = BAT.bat_sample(samples_target_weighted, OrderedResampling()).result

samples_train = unshaped.(samples_target.v)

# constructs a normalizing flow that is a chain of a `InvMulAdd` and a `Musketeer Flow`
flow = build_flow(samples_train)

# train flow 
flow_opt = optimize_flow_sequentially(samples_train, flow, Adam(1f-3))

