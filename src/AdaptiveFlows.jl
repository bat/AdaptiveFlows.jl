# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

"""
    AdaptiveFlows

A package for constructing modular normalizing flows.
"""
module AdaptiveFlows

using AffineMaps
using ArgCheck
using ArraysOfArrays
using ChangesOfVariables
using FunctionChains
using Functors
using HeterogeneousComputing
using InverseFunctions
using LinearAlgebra
using Lux
using MonotonicSplines
using Optimisers
using Random
using RecipesBase
using Statistics
using StatsFuns
using ValueShapes
using Zygote

include("adaptive_flows.jl")
include("optimize_flow.jl")
include("plotting.jl")
include("rqspline_coupling.jl")
include("utils.jl")
end # module
