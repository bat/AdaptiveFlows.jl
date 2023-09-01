# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AdaptiveFlows" begin
    include("test_adaptive_flows.jl")
    include("test_aqua.jl")
    include("test_docs.jl")
    include("test_scale_shift.jl")
    include("test_optimize_flow.jl")
    include("test_rqspline_coupling.jl")
end # testset
