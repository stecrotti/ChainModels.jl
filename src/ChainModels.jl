module ChainModels

import Base: show, length
import LogExpFunctions: logsumexp
import OffsetArrays: OffsetArray, OffsetVector
import LinearAlgebra: normalize!,  normalize
import Random: AbstractRNG
import Distributions: DiscreteMultivariateDistribution, Sampleable, Multivariate, Discrete,
    logpdf, pdf, _rand!, eltype, sampler, _logpdf, loglikelihood, mean, var, cov, entropy
import StatsBase: kldivergence
import ChainRulesCore: rrule, Tangent, NoTangent, ZeroTangent

export ChainModel, nstates,
        accumulate_left!, accumulate_right!, accumulate_left, accumulate_right,
        accumulate_middle, accumulate_middle!, logevaluate, evaluate, lognormalization,
        normalization, normalize!, normalize,
        marginals, neighbor_marginals, pair_marginals,
        loglikelihood_gradient, loglikelihood_gradient!,
        # overrides from Distributions, StatsBase
        rand, logpdf, loglikelihood, pdf, mean, var, cov, entropy,
        kldivergence
        

include("accumulators.jl")
include("chainmodel.jl")
include("overrides.jl")

end
