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

