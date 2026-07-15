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

"""
    ChainModel{T} = KChainModel{<:AbstractVector{<:AbstractArray{T,2}}}

A specialized [`KChainModel`](@ref) with K=2 nearest-neighbor interactions
"""
const ChainModel{T} = KChainModel{<:AbstractVector{<:AbstractArray{T,2}}}
function ChainModel(f::AbstractVector{<:AbstractArray{<:Real,2}})
    return KChainModel(f)
end
function ChainModel(f::AbstractVector{<:AbstractArray{<:Real,2}},
    h::AbstractVector{<:AbstractVector{<:Real}})
    return KChainModel(f, h)
end

"""
    FactorizedModel{T} = KChainModel{<:AbstractVector{<:AbstractArray{T,1}}}

A specialized [`KChainModel`](@ref) with K=1 nearest-neighbor interactions, i.e. no interactions
"""
const FactorizedModel{T} = KChainModel{<:AbstractVector{<:AbstractArray{T,1}}}
function FactorizedModel(f::AbstractVector{<:AbstractArray{<:Real,1}})
    return KChainModel(f)
end
function FactorizedModel(f::AbstractVector{<:AbstractArray{<:Real,1}},
    h::AbstractVector{<:AbstractVector{<:Real}})
    return KChainModel(f, h)
end

function KChainModel(f::AbstractVector{<:AbstractArray{<:Real,K}}, 
        h::AbstractVector{<:AbstractVector{<:Real}}) where {K}
    
    Lf = length(f) + K - 1
    L = length(h)
    L == Lf || throw(ArgumentError("Incompatible lengths, got $Lf and $L"))
    for i in eachindex(f)
        collect(size(f[i])) == length.(h[i:i+K-1]) || throw(ArgumentError("Inconsistency in sizes for f and h"))
    end
    fnew = deepcopy(f)
    for i in eachindex(h)
        a = max(1, i-L+K)
        j = min(i,lastindex(fnew))
        for xi in eachindex(h[i])
            fj_slice = selectdim(fnew[j], a, xi)
            fj_slice .+= h[i][xi]
        end
    end
    return KChainModel(fnew)
end

function rand_k_chain_model(rng::AbstractRNG, K::Integer, L::Integer, q::Integer)
    f = [randn(rng, fill(q, K)...) for _ in 1:(L-K+1)]
    return KChainModel(f)
end
function rand_k_chain_model(K::Integer, L::Integer, q::Integer)
    return rand_k_chain_model(Random.default_rng(), K, L, q)
end

"""
    rand_chain_model(rng::AbstractRNG, L::Integer, q::Integer)

Return a  [`ChainModel`](@ref) of length `L` and `q` states for each variable, with random entries
"""
rand_chain_model(rng, L, q) = rand_k_chain_model(rng, 2, L, q)
rand_chain_model(L, q) = rand_chain_model(Random.default_rng(), L, q)


"""
    rand_factorized_model(rng::AbstractRNG, L::Integer, q::Integer)

Return a  [`FactorizedModel`](@ref) of length `L` and `q` states for each variable, with random entries
"""
rand_factorized_model(rng, L, q) = rand_k_chain_model(rng, 1, L, q)
rand_factorized_model(L, q) = rand_factorized_model(Random.default_rng(), L, q)

nstates(chain::KChainModel) = nstates(chain.f)

getK(::KChainModel{<:AbstractVector{<:AbstractArray{<:Real,K}}}) where {K} = K

Base.length(chain::KChainModel) = length(chain.f) + getK(chain) - 1

accumulate_left(chain::KChainModel) = accumulate_left(chain.f)
accumulate_right(chain::KChainModel) = accumulate_right(chain.f)
accumulate_middle(chain::KChainModel) = accumulate_middle(chain.f)


function Base.show(io::IO, chain::KChainModel{T}) where {T}
    println(io, "KChainModel{$T} with $(length(chain)) variables")
end

"""
    rand_kchain_model([rng], K::Integer, L::Integer, q::Integer)

Return a  [`KChainModel`](@ref) of length `L` and `q` states for each variable, with random entries
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

function logevaluate(chain::KChainModel{<:AbstractVector{<:AbstractArray{T,K}}}, x) where {T,K}
    length(x) == length(chain) || throw(ArgumentError("x should be of same length as chain"))
    s = zero(T)
    @inbounds for i in eachindex(chain.f)
        idx = ntuple(j -> x[i + j - 1], K)
        s += chain.f[i][idx...] 
    end
    return s
end

"""
    evaluate(chain::ChainModel, x)

Evaluate the (possibly unnormalized) model at `x`
"""
evaluate(chain::KChainModel, x) = exp(logevaluate(chain, x)) 

"""
    lognormalization(chain::KChainModel; l = accumulate_left(chain))

Conceptually equivalent to `log(normalization(chain))`, less prone to numerical issues
"""
function lognormalization(chain::KChainModel; l = accumulate_left(chain)) 
    return logsumexp(last(l))
end
# special case for K=1
function lognormalization(chain::KChainModel{<:AbstractVector{<:AbstractArray{T,1}}}; 
    l = accumulate_left(chain)) where T

    return sum(logsumexp.(chain.f); init=zero(T))
end

"""
    normalization(chain::KChainModel; l = accumulate_left(chain))

Compute the normalization of the model
"""
function normalization(chain::KChainModel; l = accumulate_left(chain)) 
    return exp(lognormalization(chain; l))
end

"""
    LinearAlgebra.normalize!(chain::KChainModel; logZ = lognormalization(chain))

Normalize the distribution such that the normalization constant is equal to 1.
"""
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

function _km1_neighbor_marginals(chain::KChainModel;
    l = accumulate_left(chain), r = accumulate_right(chain))

    K = getK(chain)
    L = length(chain)

    return map(1:L-K+2) do i
        pᵢ = l[i-1] + r[i+K-1]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end

@doc raw"""
    neighbor_marginals(chain::KChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute the `K`-body marginals
```math
p(x_i, \ldots, x_{i+K-1})\quad\forall i
```
where `K` is the number of nearest-neighbors involved in the interactions.
For a `ChainModel`, it computes the pairwise marginals $p(x_i, x_{i+1}$.
Optionally pass pre-computed left and right partial normalizations.
"""
function neighbor_marginals(chain::KChainModel{<:AbstractVector{<:AbstractArray{T,K}}};
    l = accumulate_left(chain), r = accumulate_right(chain)) where {T,K}

    L = length(chain)

    return map(1:L-K+1) do i
        pᵢ = [l[i-1][x[1:end-1]...] + chain.f[i][x...] + r[i+K][x[2:end]...]
            for x in Iterators.product(axes(chain.f[i])...)]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end::Vector{Array{T,K}}
end

@doc raw"""
    nbody_neighbor_marginals(::Val{n}, chain::KChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute the `n`-body marginals
```math
p(x_i, \ldots, x_{i+n-1})\quad\forall i
```
with $n\le K$.
Optionally pass pre-computed left and right partial normalizations.
"""
function nbody_neighbor_marginals(::Val{n}, 
    chain::KChainModel{<:AbstractVector{<:AbstractArray{T,K}}}; kw...) where {n,T,K}

    0 ≤ n ≤ K || throw(ArgumentError("Expected n to be between 0 and K=$K, got $n"))
    n == K && return neighbor_marginals(chain; kw...)
    km1_marg = _km1_neighbor_marginals(chain; kw...)
    n == K-1 && return km1_marg
    L = length(chain)
    return map(1:L-n+1) do i 
        a = max(1, i-L+K-1)
        j = min(i, lastindex(km1_marg)-n+1)
        dims = (1:ndims(km1_marg[j]))[Not(a:a+n-1)]
        dropdims(sum(km1_marg[j]; dims); dims=tuple(dims...))
    end::Vector{Array{T,n}}
end

@doc raw"""
    marginals(chain::KChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute single-site marginals
```math
p(x_i)\quad\forall i
```

Optionally pass pre-computed left and right partial normalizations.
"""
function marginals(chain::KChainModel; kw...)
    return nbody_neighbor_marginals(Val(1), chain; kw...)
end

@doc raw"""
    pair_marginals(chain::ChainModel{T};
        l = accumulate_left(chain), r = accumulate_right(chain), 
        m = accumulate_middle(chain)

Compute pairwise marginals
```math
p(x_i, x_j)\quad\forall i,j
```

Optionally pass pre-computed left, right and middle partial normalizations.
"""
function pair_marginals(chain::ChainModel{T};
        l = accumulate_left(chain), r = accumulate_right(chain), 
        m = accumulate_middle(chain)) where T
    L = length(chain)
    qs = nstates(chain)
    p = [zeros(T, q1, q2) for q1 in qs, q2 in qs]
    for i in 1:L-1
        for j in i+1:L
            for xᵢ in axes(p[i,j], 1)
                for xⱼ in axes(p[i,j], 2)
                    p[i,j][xᵢ,xⱼ] = l[i-1][xᵢ] + m[i,j][xᵢ,xⱼ] + r[j+1][xⱼ]
                end
            end
            p[i,j] .-= logsumexp(p[i,j])
            p[i,j] .= exp.(p[i,j])
            p[j,i] .= p[i,j]'
        end
    end
    p
end

@doc raw"""
    avg_energy(chain::KChainModel; nmarg = neighbor_marginals(chain))

Compute the average energy 
```math
E = - \sum_{x_1, \ldots, x_L} \sum_{i=1}^{L-K+1} f_i(x_i,\ldots,x_{i+K-1}) p(x_1, \ldots, x_L)
```
"""
function avg_energy(chain::KChainModel; nmarg = neighbor_marginals(chain))
    en = 0.0
    # f = -Inf times p = 0 should give 0, so "soften" the infinite
    softinf(x) = (x == -Inf) ? -1e10 : x
    for (fᵢ,pᵢ) in zip(chain.f, nmarg)
        en -= expectation((x...)->softinf(fᵢ[x...]), pᵢ)
    end
    en
end