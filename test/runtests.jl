using ChainModels
using Test
using OffsetArrays
using InvertedIndices
using Distributions
using FiniteDifferences
using StatsBase
using Random

rng = MersenneTwister(1)
qs = (4,2,3,4,5,4)
f = [randn(rng, qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)]
chain = ChainModel(f)
pchain = ChainModel(f, Periodic)
@test (@inferred length(chain)) == (@inferred length(pchain) + 1)
@test (@inferred nstates(chain)) == qs
@test (@inferred nstates(pchain)) == qs[1:end-1]

@testset "Accumulators" begin
    include("accumulators.jl")
end

@testset "ChainModel" begin
    include("chainmodel.jl")
end

# @testset "Overrides" begin
#     include("overrides.jl")
# end

nothing