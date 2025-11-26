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

"""
    ChainModel(f::Vector{Matrix{T}}, h::Vector{Vector{T}})

Construct a ChainModel from neighboring interactions `f` and single-site biases `h`.
"""
function ChainModel(f::Vector{Matrix{T}}, h::Vector{Vector{T}}) where {T<:Real}
    Lf = length(f) + 1
    Lh = length(h)
    Lh == Lf || throw(ArgumentError("Incompatible lengths, got $Lf and $Lh"))
    for i in eachindex(f)
        sz_i, sz_ip1 = size(f[i])
        sz_i == length(h[i]) || throw(ArgumentError("Inconsistency in sizes for f and h"))
        sz_ip1 == length(h[i+1]) || throw(ArgumentError("Inconsistency in sizes for f and h"))
    end
    fnew = deepcopy(f)
    for i in eachindex(f)
        for sip1 in axes(f[i], 2)
            for si in axes(f[i], 1)
                field = ((2.0 ^ (i==1)) * h[i][si] + (2.0 ^ (i==Lf-1)) * h[i+1][sip1]) / 2
                fnew[i][si,sip1] = f[i][si,sip1] + field
            end
        end
    end
    return ChainModel(fnew)
end

"""
    rand_chain_model([rng], L::Integer, q::Integer)

Return a  `ChainModel` of length `L` and `q` states for each variable, with random entries
"""
function rand_chain_model(rng::AbstractRNG, L::Integer, q::Integer)
    f = [randn(rng, q, q) for _ in 1:(L-1)]
    return ChainModel(f)
end
function rand_chain_model(L::Integer, q::Integer)
    return rand_chain_model(Random.default_rng(), L, q)
end

accumulate_left(chain::ChainModel) = accumulate_left(chain.f)
accumulate_right(chain::ChainModel) = accumulate_right(chain.f)
accumulate_middle(chain::ChainModel) = accumulate_middle(chain.f)

"""
    Base.length(::ChainModel{T,L})

Returns `L`, the number of variables
"""
Base.length(::ChainModel{T,L}) where {T,L} = L

# treat a ChainModel as a scalar when broadcasting
Base.broadcastable(chain::ChainModel) = Ref(chain)

"""
    nstates(chain::ChainModel{T,L})

Returns a `NTuple{L,Int}` with the number of values each variable can take.
"""
nstates(chain::ChainModel{T,L}) where {T,L} = NTuple{L,Int}(nstates(chain.f))

function Base.show(io::IO, ::ChainModel{T,L}) where {T,L}
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
\begin{aligned}
Z =& \sum\limits_{x_1,\ldots,x_L}\prod\limits_{i=1}^{L-1} e^{f_i(x_i,x_{i+1})}\\
  =& \sum\limits_{x_L} e^{l_{L-1}(x_L)}
\end{aligned}
```

Optionally, pass the pre-computed left partial normalization
"""
normalization(chain::ChainModel; l = accumulate_left(chain)) = exp(lognormalization(chain; l))

@doc raw"""
    normalize!(chain::ChainModel; logZ = lognormalization(chain))

Divide each factor by $Z^{1/L}$ so that the normalization becomes $1$.
"""
function LinearAlgebra.normalize!(chain::ChainModel; logZ = lognormalization(chain))
    for fᵢ in chain.f
        fᵢ .-= logZ / length(chain.f)
    end
    chain
end

@doc raw"""
    normalize(chain::ChainModel; logZ = lognormalization(chain))

Return a new `ChainModel` equivalent to `chain` but rescaled so that normalization equal to $1$.
"""
LinearAlgebra.normalize(chain::ChainModel; logZ = lognormalization(chain)) = normalize!(deepcopy(chain); logZ)

@doc raw"""
    marginals(chain::ChainModel; l = accumulate_left(chain), r = accumulate_right(chain))

Compute single-variable marginals

```math
\begin{aligned}
p(x_i=x) &= \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x)\\
=& \frac1Z e^{l_{i-1}(x)+r_{i+1}(x)}
\end{aligned}
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

Compute nearest-neighbor marginals

```math
\begin{aligned}
p(x_i=x,x_{i+1}=x') =& \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x) \delta(x_{i+1},x')\\
=& \frac1Z e^{l_{i-1}(x) + f_i(x,x') + r_{i+2}(x')}
\end{aligned}
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
    pair_marginals(chain::ChainModel; l = accumulate_left(chain), r = accumulate_right(chain), m = accumulate_middle(chain))

Compute pair marginals

```math
\begin{aligned}
p(x_i=x,x_j=x') =& \sum_{x_1, x_2, \ldots, x_L} p(x_1, x_2, \ldots, x_L) \delta(x_i,x) \delta(x_j,x')\\
=&  \frac1Z e^{l_{i-1}(x) + m_{i,j}(x,x') + r_{j+1}(x')}
\end{aligned}
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

@doc raw"""
    avg_energy(chain::ChainModel; nmarg = neighbor_marginals(chain))

Compute the average "energy"

```math
\begin{aligned}
E =& \mathbb{E} \sum_{x_1, x_2, \ldots, x_L} \left[ -\sum\limits_{i=1}^{L-1}f_i(x_i,x_{i+1}) \right] p(x_1, x_2, \ldots, x_L)\\
=&  -\sum\limits_{i=1}^{L-1} \sum_{x,x'} f_i(x,x') p(x_i=x,x_{i+1}=x')
\end{aligned}
```

Optionally, pass the pre-computed neighbor marginals
"""
function avg_energy(chain::ChainModel; nmarg = neighbor_marginals(chain))
    en = 0.0
    for (fᵢ,pᵢ) in zip(chain.f, nmarg)
        en -= expectation((xᵢ,xᵢ₊₁)->fᵢ[xᵢ,xᵢ₊₁], pᵢ)
    end
    en
end