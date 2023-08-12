using ChainModels
using Test
using OffsetArrays
using InvertedIndices
using Distributions
using FiniteDifferences
using StatsBase
using Random

rng = MersenneTwister(0)
qs = (4,3,1,2)
f = [randn(rng, qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)]
chain = ChainModel(f)
L = length(chain)

@testset "Accumulators" begin
    include("accumulators.jl")
end

@testset "ChainModel" begin
    include("chainmodel.jl")
end

@testset "Overrides" begin
    include("overrides.jl")
end

nothing