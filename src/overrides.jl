# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoids creating a `Weight` object
function sample_noalloc(rng::AbstractRNG, w) 
    z = sum(w)
    isfinite(z) || throw(ArgumentError("Cannot sample: the set of weights has infinite sum"))
    t = rand(rng) * z
    i = 0
    cw = 0.0
    for p in w
        cw += p
        i += 1
        cw > t && return i
    end
    @assert false
end

struct ChainSampler{T,L,U} <: Sampleable{Multivariate,Discrete} where {T<:Real,L}
    chain :: ChainModel{T,L}
    r     :: OffsetVector{Matrix{T}, Vector{Matrix{T}}}

    function ChainSampler(chain::ChainModel{T,L}) where {T,L}
        U = typeof(chain)
        new{T,L,U}(chain, accumulate_right(chain))
    end
end

length(s::ChainSampler) = length(s.chain)

sampler(chain::ChainModel) = ChainSampler(chain)

function _onesample!(rng::AbstractRNG, s::ChainSampler, x::AbstractVector{<:Integer})
    (; chain, r) = s
    x[begin] = sample_noalloc(rng, exp(rx) for rx in first(r))
    for i in Iterators.drop(eachindex(x), 1)
        logp = (chain.f[i-1][x[i-1],xᵢ] + r[i+1][xᵢ] - r[i][x[i-1]] for xᵢ in eachindex(r[i+1]))
        logz = logsumexp(logp)
        x[i] = sample_noalloc(rng, exp(logpx - logz) for logpx in logp)
    end
    return x
end

function _rand!(rng::AbstractRNG, chain::ChainModel, x::AbstractVector{<:Integer})
    return _onesample!(rng, ChainSampler(chain), x)
end

function _rand!(rng::AbstractRNG, s::ChainSampler, A::DenseMatrix{<:Integer})
    return stack(_onesample!(rng, s, x) for x in eachcol(A))
end

function _logpdf(chain::ChainModel, x; logZ = lognormalization(chain)) 
    return logevaluate(chain, x) - logZ
end

function loglikelihood(chain::ChainModel{T}, x::AbstractVector{<:AbstractVector{<:Integer}}; 
        logZ = lognormalization(chain)) where T
    L = length(chain)
    all(length(xi) == L for xi in x) || throw(DimensionMismatch("inconsistent array dimensions"))
    ll = zero(T)
    for xᵃ in x
        ll += _logpdf(chain, xᵃ; logZ)
    end
    ll
end

function loglikelihood(chain::ChainModel, A::AbstractMatrix{<:Integer}; logZ = lognormalization(chain))
    return loglikelihood(chain, eachcol(A); logZ)
end

expectation(f, p::Matrix{<:Real}) = sum(f(xi,xj) * p[xi, xj] for xi in axes(p,1), xj in axes(p,2))
expectation(f, p::Vector{<:Real}) = sum(f(xi) * p[xi] for xi in eachindex(p))
expectation(p) = expectation(identity, p)

function mean(chain::ChainModel; p = marginals(chain))
    return [expectation(pᵢ) for pᵢ in p]
end

function var(chain::ChainModel; p = marginals(chain))
    return [expectation(abs2, pᵢ) - expectation(pᵢ)^2 for pᵢ in p]
end

function cov(chain::ChainModel{T}; m = marginals(chain), p = pair_marginals(chain)) where T
    L = length(chain)
    c = zeros(T, L, L)
    for i in axes(c, 1)
        pᵢ = m[i]
        c[i,i] = expectation(abs2, pᵢ) - expectation(pᵢ)^2
        for j in i+1:L
            c[i,j] = expectation(*, p[i,j]) - expectation(m[i]) * expectation(m[j])
            c[j,i] = c[i,j]
        end
    end
    c
end

function entropy(chain::ChainModel; logZ = lognormalization(chain), en = energy(chain))
    return logZ + en
end

function kldivergence(p::ChainModel, q::ChainModel; nmarg = neighbor_marginals(p))
    en = energy(p; nmarg)
    plogp = - entropy(p; en)
    plogq = 0.0
    for i in eachindex(nmarg) 
        plogq += expectation((xᵢ,xᵢ₊₁)->q.f[i][xᵢ,xᵢ₊₁], nmarg[i])
    end
    plogq -= lognormalization(q)
    # plogq = - lognormalization(q) + 
    #   sum(expectation((xᵢ,xᵢ₊₁)->q.f[i][xᵢ,xᵢ₊₁], nmarg[i]) for i in eachindex(nmarg))
    return plogp - plogq
end

"""
    loglikelihood_gradient!(df, chain::ChainModel, x; neigmarg = neighbor_marginals(chain)

In-place version of [`loglikelihood_gradient`](@ref)
"""
function loglikelihood_gradient!(df::Vector{Matrix{T}}, chain::ChainModel{T},
        x::AbstractVector{<:AbstractVector{<:Integer}}; neigmarg = neighbor_marginals(chain)) where {T}
    for dfᵢ in df
        dfᵢ .= 0
    end
    for xᵃ in x
        for i in eachindex(neigmarg)
            for (yᵢ,yᵢ₊₁) in Iterators.product(axes(df[i])...)
                df[i][yᵢ,yᵢ₊₁] += (yᵢ==xᵃ[i])*(yᵢ₊₁==xᵃ[i+1]) - neigmarg[i][yᵢ,yᵢ₊₁]
            end
        end
    end
    df
end
function loglikelihood_gradient!(df::Vector{Matrix{T}}, chain::ChainModel{T},
        A::AbstractMatrix{<:Integer}; kw...) where {T}
    return loglikelihood_gradient!(df, chain, eachcol(A); kw...)
end

@doc raw"""
    loglikelihood_gradient(chain::ChainModel, x; neigmarg = neighbor_marginals(chain))

Compute the gradient of the loglikelihood $\mathcal{L}(\boldsymbol{x})$ of samples $\{x^{(\mu)}\}_{\mu=1,\ldots,M}$ with respect to the functions `chain.f`

```math
\frac{d\mathcal{L}(\boldsymbol{x})}{d f_i(y_i,y_{i+1})} = \sum_{\mu=1}^M \delta(y_i,x_i^{(\mu)})\delta(y_{i+1},x_{i+1}^{(\mu)}) - M p(y_i,y_{i+1}) 
```

Optionally pass pre-computed neighbor marginals
"""
function loglikelihood_gradient(chain::ChainModel{T}, x;
        neigmarg = neighbor_marginals(chain)) where {T}
    loglikelihood_gradient!(deepcopy(chain.f), chain, x; neigmarg)
end

function rrule(::typeof(loglikelihood), chain::ChainModel{T,L}, x) where {T,L}
    l = accumulate_left(chain)
    r = accumulate_right(chain)
    neigmarg = neighbor_marginals(chain; l, r)
    logZ = lognormalization(chain; l)
    y = loglikelihood(chain, x; logZ)
    function loglikelihood_pullback(ȳ)
        lbar = NoTangent()
        fbar = loglikelihood_gradient(chain, x; neigmarg)
        chainbar = Tangent{ChainModel{T,L}}(; f = fbar)
        xbar = ZeroTangent()
        return lbar, chainbar, xbar
    end
    return y, loglikelihood_pullback
end