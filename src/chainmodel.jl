"""
    ChainModel{T,L} <: DiscreteMultivariateDistribution

A type to represent a discrete multivariate probability distribution factorized on a one-dimensional chain of length `L`.

## FIELDS
- `f :: Vector{Matrix{T}}` contains the factors
"""
struct ChainModel{T,L} <: DiscreteMultivariateDistribution
    f :: Vector{Matrix{T}}

    function ChainModel(f::Vector{Matrix{T}}) where {T<:Real}
        L = length(f) + 1
        all( size(f[i],2) == size(f[i+1],1) for i in 1:L-2 ) || throw(ArgumentError("Matrix sizes must be consistent"))
        return new{T,L}(f)
    end
end

accumulate_left(chain::ChainModel) = accumulate_left(chain.f)
accumulate_right(chain::ChainModel) = accumulate_right(chain.f)
accumulate_middle(chain::ChainModel) = accumulate_middle(chain.f)
accumulate_left!(chain::ChainModel) = accumulate_left!(chain.f)
accumulate_right!(chain::ChainModel) = accumulate_right!(chain.f)
accumulate_middle!(chain::ChainModel) = accumulate_middle!(chain.f)

"""
    Base.length(::ChainModel{T,L})

Returns `L`, the number of variables
"""
length(::ChainModel{T,L}) where {T,L} = L

"""
    nstates(chain::ChainModel{T,L})

Returns a `NTuple{L,Int}` with the number of values each variable can take.
"""
nstates(chain::ChainModel{T,L}) where {T,L} = NTuple{L,Int}(nstates(chain.f))

function show(io::IO, ::ChainModel{T,L}) where {T,L}
    println(io, "ChainModel{$T} with $L variables")
end

function evaluate_matrices(chain::ChainModel, x)
    (chain.f[i][x[i],x[i+1]] for i in eachindex(chain.f))
end

"""
    logevaluate(chain::ChainModel, x)

Conceptually equivalent to `log(evaluate(chain, x))`, less prone to numerical issues
"""
function logevaluate(chain::ChainModel, x)
    sum(evaluate_matrices(chain, x); init=0.0)
end

"""
    evaluate(chain::ChainModel, x)

Evaluate the (possibly unnormalized) model at `x`
"""
function evaluate(chain::ChainModel, x)
    exp(logevaluate(chain, x)) 
end

"""
    lognormalization(chain::ChainModel; l = accumulate_left(chain))

Conceptually equivalent to `log(normalization(chain))`, less prone to numerical issues
"""
lognormalization(chain::ChainModel; l = accumulate_left(chain)) = logsumexp(last(l))

@doc raw"""
    normalization(chain::ChainModel; l = accumulate_left(chain))

Compute the normalization

```math
Z = \sum\limits_{x_1,\ldots,x_L}\prod\limits_{i=1}^{L-1} e^{f_i(x_i,x_{i+1})}
```

Optionally, pass the pre-computed left partial normalization
"""
normalization(chain::ChainModel; l = accumulate_left(chain)) = exp(lognormalization(chain; l))

@doc raw"""
    normalize!(chain::ChainModel; logZ = lognormalization(chain))

Divides each factor by $Z^{1/L}$ so that the normalization becomes $1$.
"""
function normalize!(chain::ChainModel; logZ = lognormalization(chain))
    for fᵢ in chain.f
        fᵢ .-= logZ / length(chain.f)
    end
    chain
end

@doc raw"""
    normalize(chain::ChainModel; logZ = lognormalization(chain))

Return a new `ChainModel` equivalent to `chain` but rescaled so that normalization equal to $1$.
"""
normalize(chain::ChainModel; logZ = lognormalization(chain)) = normalize!(deepcopy(chain); logZ)

@doc raw"""
    marginals(chain::ChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute single-variable marginals $p(x_i=x)$

```math
p(x_i=x) = \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x)
```

Optionally, pass the pre-computed left and right partial normalization
"""
function marginals(chain::ChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)) do i
        pᵢ = [l[i-1][xᵢ] + r[i+1][xᵢ] for xᵢ in eachindex(l[i-1])]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end 

@doc raw"""
    neighbor_marginals(chain::ChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute nearest-neighbor marginals $p(x_i=x,x_{i+1}=x')$

```math
p(x_i=x,x_{i+1}=x') = \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x) \delta(x_{i+1},x')
```

Optionally, pass the pre-computed left and right partial normalization
"""
function neighbor_marginals(chain::ChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)-1) do i
        pᵢ = [l[i-1][xᵢ] + chain.f[i][xᵢ,xᵢ₊₁] + r[i+2][xᵢ₊₁] 
            for xᵢ in eachindex(l[i-1]), xᵢ₊₁ in eachindex(r[i+2])]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end

@doc raw"""
    neighbor_marginals(chain::ChainModel; l = accumulate_left(chain), r = accumulate_right(chain), m = accumulate_middle(chain))

Compute pair marginals $p(x_i=x,x_j=x')$

```math
p(x_i=x,x_j=x') = \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x) \delta(x_j,x')
```

Optionally, pass the pre-computed left, right and middle partial normalization
"""
function pair_marginals(chain::ChainModel{T};
        l = accumulate_left(chain), r = accumulate_right(chain), 
        m = accumulate_middle(chain)) where T
    L = length(chain)
    p = [zeros(T, q1, q2) for q1 in nstates(chain), q2 in nstates(chain)]
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