# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AdaptiveFlows" begin
    include("test_aqua.jl")
    include("test_adaptive_flow.jl")
    include("test_docs.jl")
end # testset
