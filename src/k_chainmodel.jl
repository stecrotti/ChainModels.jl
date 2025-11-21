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

k_accumulate_left(chain::KChainModel) = k_accumulate_left(chain.f)
k_accumulate_right(chain::KChainModel) = k_accumulate_right(chain.f)

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
    return sum(chain.f[i][x[i:i+K-1]...] for i in eachindex(chain.f); init=0.0)
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
function lognormalization(chain::KChainModel; 
    l = k_accumulate_left(chain), r = k_accumulate_right(chain)) 
    return reduce(logsumexp, last(l))

end
