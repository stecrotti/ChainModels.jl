using ChainModels
using Test
using OffsetArrays
using InvertedIndices
using Distributions
using StatsBase
using Random
using LogExpFunctions

K = 4
qs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
L = length(qs)

f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
neigmarg = neighbor_marginals(chain)
r = ChainModels.accumulate_right(chain)
l = ChainModels.accumulate_left(chain)

rng = Random.Xoshiro(0)
x = zeros(Int, L)
for _ in 1:1
    ChainModels._onesample!(rng, Distributions.sampler(chain), x)
end
# ChainModels._onesample!(rng, ChainModels.KChainSampler(chain), x)
# Distributions.rand!(rng, chain, x)

# rand(chain, 10)