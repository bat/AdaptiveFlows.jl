# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using AdaptiveFlows
using Test

using AffineMaps
using ChangesOfVariables
using DelimitedFiles
using InverseFunctions
using Random
using ValueShapes

# test input
x = readdlm("test_outputs/x.txt")
vs_test = valshape(x)
n_dims = 4

# test outputs
comp_flow_test = CompositeFlow([RQSplineCouplingModule(n_dims), RQSplineCouplingModule(n_dims)])
comp_flow_y_test = readdlm("test_outputs/comp_flow_y_test.txt")
comp_flow_ladj_test = readdlm("test_outputs/comp_flow_ladj_test.txt")
comp_flow_test_prepended = prepend_flow_module(comp_flow_test, FlowModule(InvMulAdd(ones(4), zeros(4)), false))    
comp_flow_test_appended = append_flow_module(comp_flow_test, FlowModule(InvMulAdd(ones(4), zeros(4)), false))    

@testset "CompositeFlow" begin
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(comp_flow_test, x), (comp_flow_y_test, comp_flow_ladj_test)))
    @test isapprox(comp_flow_test(x), comp_flow_y_test)
    @test comp_flow_test(vs_test) == vs_test
    
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(InverseFunctions.inverse(comp_flow_test), comp_flow_y_test), (x, .- comp_flow_ladj_test)))
    @test isapprox(InverseFunctions.inverse(comp_flow_test)(comp_flow_y_test), x)
    
    @test comp_flow_test_prepended.flow.fs[1].flow isa AffineMaps.AbstractAffineMap
    @test comp_flow_test_appended.flow.fs[end].flow isa AffineMaps.AbstractAffineMap
end
