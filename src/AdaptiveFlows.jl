# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

"""
    AdaptiveFlows

Adaptive normalizing flows.
"""
module AdaptiveFlows

using ArgCheck
using ArraysOfArrays
using ChangesOfVariables
using FunctionChains
using Functors
using HeterogeneousComputing
using InverseFunctions
using Lux
using MonotonicSplines
using Optimisers 
using Random
using StatsFuns
using ValueShapes
using Zygote

include("adaptive_flows.jl")
include("optimize_flow.jl")
include("rqspline_coupling.jl")
include("utils.jl")
end # module
