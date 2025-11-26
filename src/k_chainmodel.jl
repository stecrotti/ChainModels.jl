"""
    KChainModel{T,L} <: DiscreteMultivariateDistribution

A type to represent a discrete multivariate probability distribution factorized on a one-dimensional chain of length `L`.

## FIELDS
- `f` a vector containing the factors as arrays
"""
struct KChainModel{T<:AbstractVector{<:AbstractArray{<:Real}}} <: DiscreteMultivariateDistribution
    f :: T

    function KChainModel(f::AbstractVector{<:AbstractArray{<:Real,K}}) where {K}
        all(1:length(f)-1) do i 
            size(f[i])[2:end] == size(f[i+1])[1:end-1]
        end || throw(ArgumentError("Matrix sizes must be consistent"))
        return new{typeof(f)}(f)
    end
end

getK(::KChainModel{<:AbstractVector{<:AbstractArray{<:Real,K}}}) where {K} = K

Base.length(chain::KChainModel) = length(chain.f) + getK(chain) - 1

accumulate_left(chain::KChainModel) = k_accumulate_left(chain.f)
accumulate_right(chain::KChainModel) = k_accumulate_right(chain.f)

"""
    rand_kchain_model([rng], K::Integer, L::Integer, q::Integer)

Return a  `KChainModel` of length `L` and `q` states for each variable, with random entries
"""
function rand_kchain_model(rng::AbstractRNG, K::Integer, L::Integer, q::Integer)
    f = [randn(rng, fill(q, K)...) for _ in 1:(L-K+1)]
    return KChainModel(f)
end
function rand_kchain_model(K::Integer, L::Integer, q::Integer)
    return rand_kchain_model(Random.default_rng(), K, L, q)
end

# treat a KChainModel as a scalar when broadcasting
Base.broadcastable(chain::KChainModel) = Ref(chain)

function evaluate_factors(chain::KChainModel, x)
    K = getK(chain)
    return (chain.f[i][x[i:i+K-1]...] for i in eachindex(chain.f))
end

function logevaluate(chain::KChainModel, x)
    length(x) == length(chain) || throw(ArgumentError("x should be of same length as chain"))
    K = getK(chain)
    return @views sum(chain.f[i][x[i:i+K-1]...] for i in eachindex(chain.f); init=0.0)
end

"""
    evaluate(chain::ChainModel, x)

Evaluate the (possibly unnormalized) model at `x`
"""
evaluate(chain::KChainModel, x) = exp(logevaluate(chain, x)) 

"""
    lognormalization(chain::ChainModel; l = accumulate_left(chain))

Conceptually equivalent to `log(normalization(chain))`, less prone to numerical issues
"""
function lognormalization(chain::KChainModel; l = accumulate_left(chain)) 
    return reduce(logsumexp, last(l))
end

function normalization(chain::KChainModel; l = accumulate_left(chain)) 
    return exp(lognormalization(chain; l))
end

function LinearAlgebra.normalize!(chain::KChainModel; 
    logZ = lognormalization(chain))

    for fᵢ in chain.f
        fᵢ .-= logZ / length(chain.f)
    end
    return chain
end

function LinearAlgebra.normalize(chain::KChainModel; 
    logZ = lognormalization(chain))
    
    return normalize!(deepcopy(chain); logZ)
end

function Km1_neighbor_marginals(chain::KChainModel;
    l = accumulate_left(chain), r = accumulate_right(chain))

    K = getK(chain)
    L = length(chain)

    return map(1:L-K+2) do i
        pᵢ = l[i-1] + r[i+K-1]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end

function neighbor_marginals(chain::KChainModel;
    l = accumulate_left(chain), r = accumulate_right(chain))

    K = getK(chain)
    L = length(chain)

    return map(1:L-K+1) do i
        pᵢ = [l[i-1][x[1:end-1]...] + chain.f[i][x...] + r[i+K][x[2:end]...]
            for x in Iterators.product(axes(chain.f[i])...)]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end

function nbody_marginals(n::Integer, chain::KChainModel; kw...)
end

function marginals(chain::KChainModel; kw...)
    km1_marg = Km1_neighbor_marginals(chain; kw...)

    K = getK(chain)
    L = length(chain)

    return map(1:length(chain)) do i 
        a = 1 + max(0, i - (L-K+2))
        j = min(i,lastindex(km1_marg))
        dims = (1:ndims(km1_marg[j]))[Not(a)]
        vec(sum(km1_marg[j]; dims))
    end
end

function energy(chain::KChainModel; nmarg = neighbor_marginals(chain))
    en = 0.0
    for (fᵢ,pᵢ) in zip(chain.f, nmarg)
        en -= expectation((x...)->fᵢ[x...], pᵢ)
    end
    en
end