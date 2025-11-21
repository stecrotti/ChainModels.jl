module ChainModels

using LogExpFunctions: logsumexp
using OffsetArrays: OffsetArray, OffsetVector
using LinearAlgebra: LinearAlgebra, normalize, normalize!
using Random: Random, AbstractRNG, rand
using Distributions: Distributions, DiscreteMultivariateDistribution, 
    Sampleable, Multivariate, Discrete, logpdf, pdf
using StatsBase: StatsBase, mean, var, cov, entropy, kldivergence, loglikelihood

export ChainModel, rand_chain_model, nstates,
        accumulate_left!, accumulate_right!, accumulate_left, accumulate_right,
        accumulate_middle, accumulate_middle!, logevaluate, evaluate, lognormalization,
        normalization, normalize!, normalize,
        marginals, neighbor_marginals, pair_marginals, energy
        # overrides from Distributions, StatsBase
        rand, mean, var, cov, entropy,
        kldivergence, loglikelihood, logpdf, pdf
        

include("accumulators.jl")
include("chainmodel.jl")
include("overrides.jl")
include("k_chainmodel.jl")

export KChainModel, rand_kchain_model, k_accumulate_left, k_accumulate_left!,
    k_accumulate_right, k_accumulate_right!

end
