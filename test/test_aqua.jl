# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AdaptiveFlows

# ToDo: Fix ambiguities and enable ambiguity testing:
#=
Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(AdaptiveFlows))
end 
=#

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        AdaptiveFlows,
        ambiguities = false,
        unbound_args = false
    )
end
