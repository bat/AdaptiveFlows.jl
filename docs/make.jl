# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using AdaptiveFlows

# Doctest setup
DocMeta.setdocmeta!(
    AdaptiveFlows,
    :DocTestSetup,
    :(using AdaptiveFlows);
    recursive=true,
)

makedocs(
    sitename = "AdaptiveFlows",
    modules = [AdaptiveFlows],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/AdaptiveFlows.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/bat/AdaptiveFlows.jl.git",
    forcepush = true,
    push_preview = true,
)
