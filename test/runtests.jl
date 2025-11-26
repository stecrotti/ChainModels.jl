using ChainModels
using Test
using OffsetArrays
using InvertedIndices
using Distributions
using StatsBase
using Random
using LogExpFunctions


Ks = 1:4
qs = (2,3,4,5,6)
L = length(qs)

@testset "KChainModel" begin
    include("k_chainmodel.jl")
end

@testset "Overrides" begin
    include("overrides.jl")
end

@testset "ChainModel" begin
    include("chainmodel.jl")
end

nothing