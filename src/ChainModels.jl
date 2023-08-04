module ChainModels

import Base: show, length
import OffsetArrays: OffsetArray, OffsetVector
import LinearAlgebra: mul!
import Random: AbstractRNG
import Distributions: DiscreteMultivariateDistribution, Sampleable, Multivariate, Discrete, 
    logpdf, pdf, _rand!, eltype, sampler, _logpdf, mean, var, cov, entropy

export AbstractChainModel, ChainModel, nstates,
        accumulate_left!, accumulate_right!, accumulate_left,
        accumulate_right, accumulate_middle, evaluate, normalization,
        marginals, neighbor_marginals, pair_marginals,
        # overrides from Distributions.jl
        logpdf, pdf, mean, var, cov, entropy


include("accumulators.jl")
include("chainmodel.jl")
include("overrides.jl")

end
