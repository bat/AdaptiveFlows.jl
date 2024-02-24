# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using Test

using ArraysOfArrays
using ChangesOfVariables
using DelimitedFiles
using HeterogeneousComputing
using InverseFunctions
using LinearAlgebra
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

# test losses and gradients
sn_logpdf = AdaptiveFlows.PushForwardLogDensity(InvMulAdd(I(n_dims), zeros(n_dims)), AdaptiveFlows.std_normal_logpdf)
sn_logds = sn_logpdf.logdensity(x)

negll_rqs_comp_flow_test, negll_rqs_comp_flow_grad = negll_flow(composite_flow_test, x, [1,1], (sn_logpdf, sn_logpdf))
negll_rqs_cb_test, negll_rqs_cb_test_grad = negll_flow(rqs_cb_test, x, [1,1], (sn_logpdf, sn_logpdf))
negll_rqs_cm_test, negll_rqs_cm_test_grad = negll_flow(musketeer_flow, x, [1,1], (sn_logpdf, sn_logpdf))

KLDiv_rqs_comp_flow_test, KLDiv_rqs_comp_flow_grad = KLDiv_flow(composite_flow_test, x, sn_logds, (sn_logpdf, sn_logpdf))
KLDiv_rqs_cb_test, KLDiv_rqs_cb_test_grad = KLDiv_flow(rqs_cb_test, x, sn_logds, (sn_logpdf, sn_logpdf))
KLDiv_rqs_cm_test, KLDiv_rqs_cm_test_grad = KLDiv_flow(musketeer_flow, x, sn_logds, (sn_logpdf, sn_logpdf))


# test optimized flows and opt states
logpdfs = (AdaptiveFlows.std_normal_logpdf, AdaptiveFlows.std_normal_logpdf)

res_comp_flow_forw, opt_state_comp_flow_forw, loss_hist_comp_flow_forw = optimize_flow(smpls, composite_flow_test, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cb_forw, opt_state_rqs_cb_forw, loss_hist_rqs_cb_forw = optimize_flow(smpls, rqs_cb_test, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cm_forw, opt_state_rqs_cm_forw, loss_hist_rqs_cm_forw = optimize_flow(smpls, musketeer_flow, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)

res_comp_flow_forw_seq, opt_state_comp_flow_forw_seq, loss_hist_comp_flow_forw_seq = optimize_flow(smpls, composite_flow_test, Adam(3f-4); sequential = true, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cm_forw_seq, opt_state_rqs_cm_forw_seq, loss_hist_rqs_cm_forw_seq = optimize_flow(smpls, musketeer_flow, Adam(3f-4); sequential = true, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)

res_comp_flow_inv, opt_state_comp_flow_inv, loss_hist_comp_flow_inv = optimize_flow(smpls, composite_flow_test, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cb_inv, opt_state_rqs_cb_inv, loss_hist_rqs_cb_inv = optimize_flow(smpls, rqs_cb_test, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cm_inv, opt_state_rqs_cm_inv, loss_hist_rqs_cm_inv = optimize_flow(smpls, musketeer_flow, Adam(3f-4); sequential = false, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)

res_comp_flow_inv_seq, opt_state_comp_flow_inv_seq, loss_hist_comp_flow_inv_seq = optimize_flow(smpls, composite_flow_test, Adam(3f-4); sequential = true, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)
res_rqs_cm_inv_seq, opt_state_rqs_cm_inv_seq, loss_hist_rqs_cm_inv_seq = optimize_flow(smpls, musketeer_flow, Adam(3f-4); sequential = true, loss = AdaptiveFlows.negll_flow, logpdf = logpdfs, nbatches = 1, nepochs = 10)

y_normalized_test = readdlm("test_outputs/y_normalized_test.txt")

# Currently only tests the value of the losses the types of the gradients
@testset "losses and gradients" begin
       @test isapprox(AdaptiveFlows.std_normal_logpdf(1), -1.4189385332046727)    
       @test isapprox(AdaptiveFlows.std_normal_logpdf([1 1]), [-1.4189385332046727, -1.4189385332046727]) 

       @test (isapprox(negll_rqs_comp_flow_test, 10.981005962569172) &&
              negll_rqs_comp_flow_grad isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}}}}}}}}}}}})
       @test (isapprox(negll_rqs_cm_test, 9.089891518469202) && 
              negll_rqs_cm_test_grad isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}}}}}}})
       @test (isapprox(negll_rqs_cb_test, 8.477053344225029) &&
              negll_rqs_cb_test_grad isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}})

       @test (isapprox(KLDiv_rqs_comp_flow_test, 0.11109861572133527) &&
              KLDiv_rqs_comp_flow_grad isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}}}}}}}}}}}})
       @test (isapprox(KLDiv_rqs_cb_test, 0.0005145947143370143) &&
              KLDiv_rqs_cb_test_grad isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}})
       @test (isapprox(KLDiv_rqs_cm_test, 0.009618386067154767) &&
              KLDiv_rqs_cm_test_grad isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Nothing, Nothing, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}, NamedTuple{(:weight, :bias), Tuple{Matrix{Float32}, Matrix{Float32}}}}}, Nothing}}}}}}})
end

@testset "optimize_flow" begin
       @test (typeof(res_comp_flow_forw) == typeof(composite_flow_test) &&
              opt_state_comp_flow_forw isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}}}}}}})
       @test (typeof(res_rqs_cb_forw) == typeof(rqs_cb_test) &&
              opt_state_rqs_cb_forw isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_rqs_cm_forw) == typeof(musketeer_flow) &&
              opt_state_rqs_cm_forw isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}})    
       
       @test (typeof(res_comp_flow_forw_seq) == typeof(composite_flow_test) &&
              opt_state_comp_flow_forw_seq[1][1] isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_rqs_cm_forw_seq) == typeof(musketeer_flow) &&
              opt_state_rqs_cm_forw_seq[1] isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
   
       @test (typeof(res_comp_flow_inv) == typeof(composite_flow_test) &&
              opt_state_comp_flow_inv isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}}}}}}})
       @test (typeof(res_rqs_cb_inv) == typeof(rqs_cb_test) &&
              opt_state_rqs_cb_inv isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_rqs_cm_inv) == typeof(musketeer_flow) &&
              opt_state_rqs_cm_inv isa NamedTuple{(:flow,), Tuple{NamedTuple{(:fs,), Tuple{Vector{NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}}}}}}})    
       
       @test (typeof(res_comp_flow_inv_seq) == typeof(composite_flow_test) &&
              opt_state_comp_flow_inv_seq[1][1] isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
       @test (typeof(res_rqs_cm_inv_seq) == typeof(musketeer_flow) &&
              opt_state_rqs_cm_inv_seq[1] isa NamedTuple{(:mask, :nn, :nn_parameters, :nn_state), Tuple{Tuple{}, NamedTuple{(:layers,), Tuple{NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{Tuple{}, Tuple{}, Tuple{}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}, NamedTuple{(:weight, :bias), Tuple{Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}, Optimisers.Leaf{Adam, Tuple{Matrix{Float32}, Matrix{Float32}, Tuple{Float32, Float32}}}}}}}, NamedTuple{(:layer_1, :layer_2, :layer_3), Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}}}})
end
