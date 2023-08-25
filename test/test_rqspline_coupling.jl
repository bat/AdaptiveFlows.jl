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
# rqs_cm = rational  quadratic coupling module

# Test inputs
n_dims = 4
n_smpls = 100

rng = MersenneTwister(1234)
x = randn(rng, n_dims, n_smpls)
smpls = nestedview(x);

vs_test = valshape(x)

# Test outputs
# rqs_cb_y_test, rqs_cb_ladj_test = ChangesOfVariables.with_logabsdet_jacobian(rqs_cb_test, x)
rqs_cb_y_test = readdlm("test_outputs/rqs_cb_y_test.txt")
rqs_cb_ladj_test = readdlm("test_outputs/rqs_cb_ladj_test.txt")

# rqs_cm_y_test, rqs_cm_ladj_test = ChangesOfVariables.with_logabsdet_jacobian(rqs_cm_test_musk, x)
rqs_cm_y_test = readdlm("test_outputs/rqs_cm_y_test.txt")
rqs_cm_ladj_test = readdlm("test_outputs/rqs_cm_ladj_test.txt")

# Test instances of structs 
test_dims_tt = [1]

rqs_cb_mask = fill(false, n_dims)
rqs_cb_mask[test_dims_tt...] = true
rqs_cb_nn = AdaptiveFlows.get_neural_net(n_dims - sum(rqs_cb_mask), 29)
rqs_cb_compute_unit = CPUnit()

rqs_cb_test = RQSplineCouplingBlock(rqs_cb_mask, rqs_cb_nn, rqs_cb_compute_unit)
rqs_inv_cb_test = InverseRQSplineCouplingBlock(rqs_cb_mask, rqs_cb_nn, rqs_cb_compute_unit)


rqs_cm_test_direct = RQSplineCouplingModule(fill(rqs_cb_test, 4))
rqs_cm_test_vbte = RQSplineCouplingModule(n_dims, [1, 2, 3:4])
rqs_cm_test_musk = RQSplineCouplingModule(n_dims)
rqs_cm_test_2 = RQSplineCouplingModule(n_dims, 2)

rqs_cm_test_inv_musk = InverseFunctions.inverse(rqs_cm_test_musk)


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
    @test rqs_cm_test_direct.flow_module isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}}
    @test (rqs_cm_test_vbte.flow_module isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
       length(rqs_cm_test_vbte.flow_module.fs) == 3 && 
       rqs_cm_test_vbte.flow_module.fs[1].nn == rqs_cb_test.nn &&
       rqs_cm_test_vbte.flow_module.fs[end].nn.layers[1].in_dims == n_dims - 2)
    @test (rqs_cm_test_musk.flow_module isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
           length(rqs_cm_test_musk.flow_module.fs) == n_dims && 
           rqs_cm_test_musk.flow_module.fs[1].nn == rqs_cb_test.nn)
    @test (rqs_cm_test_2.flow_module isa FunctionChains.FunctionChain{Vector{RQSplineCouplingBlock}} && 
       length(rqs_cm_test_2.flow_module.fs) == ceil(n_dims / 2))
    
    @test rqs_cm_test_inv_musk.flow_module isa FunctionChain{Vector{InverseRQSplineCouplingBlock}}
    
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(rqs_cm_test_musk, x), (rqs_cm_y_test, rqs_cm_ladj_test)))
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(rqs_cm_test_inv_musk, rqs_cm_y_test), (x, .- rqs_cm_ladj_test)))
    
    @test isapprox(rqs_cm_test_musk(x), rqs_cm_y_test)
    @test rqs_cm_test_musk(vs_test) == vs_test
    
    @test isapprox(rqs_cm_test_inv_musk(rqs_cm_y_test), x)
    @test rqs_cm_test_inv_musk(vs_test) == vs_test
end
