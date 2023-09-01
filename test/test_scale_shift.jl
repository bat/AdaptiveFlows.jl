# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using AdaptiveFlows
using Test

using ArraysOfArrays
using InverseFunctions
using LinearAlgebra
using Random
using Statistics
using ValueShapes

# test inputs 
n_dims = 4
n_smpls = 10

rng = MersenneTwister(1234)
x = muladd(Diagonal(randn(rng, n_dims)), randn(rng, n_dims, n_smpls), randn(rng, n_dims))

smpls = nestedview(x)
vs_test = valshape(x)

scale_shift_test = ScaleShiftModule(x)
inv_scale_shift_test = inverse(scale_shift_test)

y_test = scale_shift_test(x)
x_inverted_test = inv_scale_shift_test(y_test)

stds_test = vec(std(y_test, dims = 2))
means_test = vec(mean(y_test, dims = 2))

@testset "ScaleShiftModule" begin
    @test all(isapprox.(stds_test, 1)) && all(isapprox.(means_test, 0, atol = 1f-15))
    @test all(isapprox.(x_inverted_test, x))
    
    @test scale_shift_test(vs_test) == vs_test
    @test all(isapprox.(with_logabsdet_jacobian(scale_shift_test, x)[2], 10.637223371435223))
end
