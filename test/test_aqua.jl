# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AdaptiveFlows

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(AdaptiveFlows))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        AdaptiveFlows,
        ambiguities = false,
        project_toml_formatting = VERSION≥v"1.7"
    )
end # testset
