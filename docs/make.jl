using Documenter
using ChainModels

makedocs(
    sitename = "ChainModels",
    format = Documenter.HTML(),
    modules = [ChainModels],
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ],
    push_preview = true
)

deploydocs(
    repo = "github.com/stecrotti/ChainModels.jl.git",
)
