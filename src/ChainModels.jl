module ChainModels

using LogExpFunctions: logsumexp
using OffsetArrays: OffsetArray, OffsetVector
using LinearAlgebra: LinearAlgebra, normalize, normalize!
using Random: Random, AbstractRNG, rand
using Distributions: Distributions, DiscreteMultivariateDistribution, 
    Sampleable, Multivariate, Discrete, logpdf, pdf
using StatsBase: StatsBase, mean, var, cov, entropy, kldivergence, loglikelihood
using InvertedIndices: Not

export KChainModel, ChainModel, 
        rand_k_chain_model, rand_chain_model, rand_factorized_model, nstates, getK,
        accumulate_left!, accumulate_right!, accumulate_left, accumulate_right,
        accumulate_middle, accumulate_middle!, logevaluate, evaluate, lognormalization,
        normalization, normalize!, normalize,
        marginals, nbody_neighbor_marginals, neighbor_marginals, pair_marginals, avg_energy
        # overrides from Distributions, StatsBase
        rand, mean, var, cov, entropy,
        kldivergence, loglikelihood, logpdf, pdf
        

include("accumulators.jl")
include("k_chainmodel.jl")
include("overrides.jl")

end
