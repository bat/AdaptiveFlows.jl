# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using AdaptiveFlows
using Test

using ChangesOfVariables
using DelimitedFiles
using InverseFunctions
using Random
using ValueShapes

# test inputs 
n_dims = 4
n_smpls = 10

rng = MersenneTwister(1234)
x = randn(rng, n_dims, n_smpls)

vs_test = valshape(x)

comp_flow_test = CompositeFlow([RQSplineCouplingModule(4), RQSplineCouplingModule(4)])
prepended_flow_test = prepend_flow_module(comp_flow_test, ScaleShiftModule(ones(4), zeros(4)))    
appended_flow_test = append_flow_module(comp_flow_test, ScaleShiftModule(ones(4), zeros(4))) 

# test outputs 
# comp_flow_y_test, comp_flow_ladj_test = with_logabsdet_jacobian(comp_flow_test, x)
comp_flow_y_test = readdlm("test_outputs/comp_flow_y_test.txt")
comp_flow_ladj_test = readdlm("test_outputs/comp_flow_ladj_test.txt")

@testset "CompositeFlow" begin
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(comp_flow_test, x), (comp_flow_y_test, comp_flow_ladj_test)))
    @test isapprox(comp_flow_test(x), comp_flow_y_test)
    @test comp_flow_test(vs_test) == vs_test
    
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(InverseFunctions.inverse(comp_flow_test), comp_flow_y_test), (x, .- comp_flow_ladj_test)))
    @test isapprox(InverseFunctions.inverse(comp_flow_test)(comp_flow_y_test), x)

    @test prepended_flow_test.flow.fs[1] isa ScaleShiftModule   
    @test appended_flow_test.flow.fs[end] isa ScaleShiftModule
end
