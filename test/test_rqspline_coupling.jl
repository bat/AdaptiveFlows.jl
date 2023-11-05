# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using Test

using ArraysOfArrays
using ChangesOfVariables
using DelimitedFiles
using FunctionChains
using HeterogeneousComputing
using InverseFunctions
using Lux
using Optimisers
using Random
using ValueShapes

# rqs_cb = rational quadratic coupling block
# rqs_cm = rational quadratic coupling module

# Test inputs
x = readdlm("test_outputs/x.txt")
n_dims = 4

smpls = nestedview(x)
vs_test = valshape(x)

# Low level structs
test_dims_tt = [1]

rqs_cb_mask = fill(false, n_dims)
rqs_cb_mask[test_dims_tt...] = true
rqs_cb_nn = AdaptiveFlows.get_neural_net(n_dims - sum(rqs_cb_mask), 29)
rqs_cb_compute_unit = CPUnit()

rqs_cb_test = RQSplineCouplingBlock(rqs_cb_mask, rqs_cb_nn, rqs_cb_compute_unit)
rqs_inv_cb_test = InverseRQSplineCouplingBlock(rqs_cb_mask, rqs_cb_nn, rqs_cb_compute_unit)

# rqs_cb_y_test, rqs_cb_ladj_test = ChangesOfVariables.with_logabsdet_jacobian(rqs_cb_test, x)
rqs_cb_y_test = readdlm("test_outputs/rqs_cb_y_test.txt")
rqs_cb_ladj_test = readdlm("test_outputs/rqs_cb_ladj_test.txt")

# test construction of rqs coupling modules
rqs_cm_test_direct = RQSplineCouplingModule(fill(rqs_cb_test, 4))
rqs_cm_test_vector_block_target_els = RQSplineCouplingModule(n_dims, [1, 2, 3:4])

# full musketeer flow
musketeer_flow = RQSplineCouplingModule(n_dims)

#rqs cm flow with two blocks, transforming two dimensions each
rqs_cm_test_2 = RQSplineCouplingModule(n_dims, 2)

musketeer_flow_inv = InverseFunctions.inverse(musketeer_flow)

musketeer_y_test = readdlm("test_outputs/musketeer_y_test.txt")
musketeer_ladj_test = readdlm("test_outputs/musketeer_ladj_test.txt")

@testset "RQSplineCouplingBlock" begin
    @test (rqs_cb_test isa RQSplineCouplingBlock && rqs_cb_test.mask == rqs_cb_mask && 
           rqs_cb_test.nn == rqs_cb_nn && 
           rqs_cb_test.nn_parameters isa NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}} && 
           rqs_cb_test.nn_state isa NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}} )
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(rqs_cb_test, x), (rqs_cb_y_test, rqs_cb_ladj_test)))
    @test isapprox(rqs_cb_test(x), rqs_cb_y_test)
    @test rqs_cb_test(vs_test) == vs_test
    @test InverseFunctions.inverse(rqs_cb_test) == InverseRQSplineCouplingBlock(rqs_cb_test.mask, rqs_cb_test.nn, rqs_cb_test.nn_parameters, rqs_cb_test.nn_state)
end

@testset "InverseRQSplineCouplingBlock" begin
    @test (rqs_inv_cb_test isa InverseRQSplineCouplingBlock && rqs_inv_cb_test.mask == rqs_cb_mask && 
           rqs_inv_cb_test.nn == rqs_cb_nn && 
           rqs_inv_cb_test.nn_parameters isa NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}} && 
           rqs_inv_cb_test.nn_state isa NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}} )
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(rqs_inv_cb_test, rqs_cb_y_test), (x, .- rqs_cb_ladj_test)))
    @test isapprox(rqs_inv_cb_test(rqs_cb_y_test), x)
    @test rqs_inv_cb_test(vs_test) == vs_test
    @test InverseFunctions.inverse(rqs_inv_cb_test) == RQSplineCouplingBlock(rqs_inv_cb_test.mask, rqs_inv_cb_test.nn, rqs_inv_cb_test.nn_parameters, rqs_inv_cb_test.nn_state)    
end

@testset "apply_rqs_coupling_flow" begin
    @test all(isapprox.(apply_rqs_coupling_flow(rqs_cb_test, x), (rqs_cb_y_test, rqs_cb_ladj_test)))
    @test all(isapprox.(apply_rqs_coupling_flow(rqs_inv_cb_test, rqs_cb_y_test), (x, .- rqs_cb_ladj_test)))    
end

@testset "RQSplineCouplingModule" begin
    @test rqs_cm_test_direct.flow isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}}
    @test (rqs_cm_test_vector_block_target_els.flow isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
       length(rqs_cm_test_vector_block_target_els.flow.fs) == 3 && 
       rqs_cm_test_vector_block_target_els.flow.fs[1].nn == rqs_cb_test.nn &&
       rqs_cm_test_vector_block_target_els.flow.fs[end].nn.layers[1].in_dims == n_dims - 2)
    @test (musketeer_flow.flow isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
           length(musketeer_flow.flow.fs) == n_dims && 
           musketeer_flow.flow.fs[1].nn == rqs_cb_test.nn)
    @test (rqs_cm_test_2.flow isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
       length(rqs_cm_test_2.flow.fs) == ceil(n_dims / 2))
    
    @test musketeer_flow_inv.flow isa FunctionChain{Vector{InverseRQSplineCouplingBlock}}
    
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(musketeer_flow, x), (musketeer_y_test, musketeer_ladj_test)))
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(musketeer_flow_inv, musketeer_y_test), (x, .- musketeer_ladj_test)))
    
    @test isapprox(musketeer_flow(x), musketeer_y_test)
    @test musketeer_flow(vs_test) == vs_test
    
    @test isapprox(musketeer_flow_inv(musketeer_y_test), x)
    @test musketeer_flow_inv(vs_test) == vs_test
end
