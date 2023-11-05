# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using Test

using ArraysOfArrays
using ChangesOfVariables
using DelimitedFiles
using HeterogeneousComputing
using InverseFunctions
using Optimisers
using Random
using ValueShapes

# Test inputs
x = readdlm("test_outputs/x.txt")
smpls = nestedview(x)
n_dims = 4

composite_flow_test = CompositeFlow([RQSplineCouplingModule(n_dims), RQSplineCouplingModule(n_dims)])

test_dims_tt = [1]
rqs_cb_mask = fill(false, n_dims)
rqs_cb_mask[test_dims_tt...] = true
rqs_cb_nn = AdaptiveFlows.get_neural_net(n_dims - sum(rqs_cb_mask), 29)
rqs_cb_compute_unit = CPUnit()
rqs_cb_test = RQSplineCouplingBlock(rqs_cb_mask, rqs_cb_nn, rqs_cb_compute_unit)

musketeer_flow = RQSplineCouplingModule(n_dims)

# test negll and grads
negll_comp_flow_test, rqs_comp_flow_grad = mvnormal_negll_flow_grad(composite_flow_test, x)
negll_rqs_cb_test, rqs_cb_test_grad = mvnormal_negll_flow_grad(rqs_cb_test, x)
negll_rqs_cm_test, rqs_cm_test_grad = mvnormal_negll_flow_grad(musketeer_flow, x)

# test optimized flows and opt states
res_comp_flow, opt_state_comp_flow, negll_hist_comp_flow = optimize_flow(smpls, composite_flow_test, Adam(3f-4); nbatches = 1, nepochs = 100)
res_rqs_cb, opt_state_rqs_cb, negll_hist_rqs_cb = optimize_flow(smpls, rqs_cb_test, Adam(3f-4); nbatches = 1, nepochs = 100)
res_rqs_cm, opt_state_rqs_cm, negll_hist_rqs_cm = optimize_flow(smpls, musketeer_flow, Adam(3f-4); nbatches = 1, nepochs = 100)

res_comp_flow_seq, opt_state_comp_flow_seq, negll_hist_comp_flow_seq = optimize_flow_sequentially(smpls, composite_flow_test, Adam(3f-4); nbatches = 1, nepochs = 100)
res_musketeer_seq, opt_state_rqs_cm_seq, negll_hist_rqs_cm_seq = optimize_flow(smpls, musketeer_flow, Adam(3f-4); nbatches = 1, nepochs = 100)

y_normalized_test = readdlm("test_outputs/y_normalized_test.txt")

@testset "negll and grad" begin
       @test isapprox(AdaptiveFlows.std_normal_logpdf(1), -1.4189385332046727)    
       
       @test isapprox(mvnormal_negll_flow(composite_flow_test, x), 10.981005962569172)
       @test isapprox(mvnormal_negll_flow(musketeer_flow, x), 9.089891518469202)
       @test isapprox(mvnormal_negll_flow(rqs_cb_test, x), 2.289609039464419)
       
       @test (isapprox(negll_comp_flow_test, 10.981005962569172) &&
              rqs_cb_test_grad isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}})
       @test (isapprox(negll_rqs_cm_test, 9.089891518469202) && 
              rqs_cm_test_grad isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}}}}}}})
       @test (isapprox(negll_rqs_cb_test, 2.289609039464419) &&
              rqs_cb_test_grad isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}})
end

@testset "optimize_flow" begin
       @test (typeof(res_comp_flow) == typeof(composite_flow_test) &&
              opt_state_comp_flow isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}}}}}}})
       @test (typeof(res_rqs_cb) == typeof(rqs_cb_test) &&
              opt_state_rqs_cb isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_rqs_cm) == typeof(musketeer_flow) &&
              opt_state_rqs_cm isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}})    
       
       @test (typeof(res_comp_flow_seq) == typeof(composite_flow_test) &&
              opt_state_comp_flow_seq[1][1] isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float64, Float64}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_musketeer_seq) == typeof(musketeer_flow) &&
              opt_state_rqs_cm_seq[1][1] isa Array{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}, Optimisers.Leaf{Adam, Tuple{Array{Float32, 2}, Array{Float32, 2}, Tuple{Float64, Float64}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}, 1})

       @test isapprox(res_musketeer_seq(x), y_normalized_test, atol = 1e-5)
end
