# This file is a part of AdaptiveFlows.jl, licensed under the MIT License (MIT).

using Test
using AdaptiveFlows
import Documenter

Documenter.DocMeta.setdocmeta!(
    AdaptiveFlows,
    :DocTestSetup,
    :(using AdaptiveFlows);
    recursive=true,
)
Documenter.doctest(AdaptiveFlows)
