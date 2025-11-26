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
end

struct ChainSampler{T,L,U} <: Sampleable{Multivariate,Discrete} where {T<:Real,L}
    chain :: ChainModel{T,L}
    r     :: OffsetVector{Matrix{T}, Vector{Matrix{T}}}

    function ChainSampler(chain::ChainModel{T,L}) where {T,L}
        U = typeof(chain)
        new{T,L,U}(chain, accumulate_right(chain))
    end
end

Base.length(s::ChainSampler) = length(s.chain)

Distributions.sampler(chain::ChainModel) = ChainSampler(chain)

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

function Distributions._rand!(rng::AbstractRNG, chain::ChainModel, x::AbstractVector{<:Integer})
    return _onesample!(rng, ChainSampler(chain), x)
end

function Distributions._rand!(rng::AbstractRNG, s::ChainSampler, A::DenseMatrix{<:Integer})
    return stack(_onesample!(rng, s, x) for x in eachcol(A))
end

function Distributions._logpdf(chain::ChainModel, x; logZ = lognormalization(chain)) 
    return logevaluate(chain, x) - logZ
end

expectation(f, p::AbstractArray{<:Real}) = sum(f(x...) * p[x...] for x in Iterators.product(axes(p)...))
expectation(f, p::Matrix{<:Real}) = sum(f(xi,xj) * p[xi, xj] for xi in axes(p,1), xj in axes(p,2))
expectation(f, p::Vector{<:Real}) = sum(f(xi) * p[xi] for xi in eachindex(p))
expectation(p) = expectation(identity, p)


function StatsBase.mean(chain::ChainModel; p = marginals(chain))
    return [expectation(pᵢ) for pᵢ in p]
end

function StatsBase.var(chain::ChainModel; p = marginals(chain))
    return [expectation(abs2, pᵢ) - expectation(pᵢ)^2 for pᵢ in p]
end

function StatsBase.cov(chain::ChainModel{T}; m = marginals(chain), p = pair_marginals(chain)) where T
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

function StatsBase.entropy(chain::ChainModel; 
    logZ = lognormalization(chain), en = avg_energy(chain))

    return logZ + en
end

function StatsBase.kldivergence(p::ChainModel, q::ChainModel; nmarg = neighbor_marginals(p))
    en = avg_energy(p; nmarg)
    plogp = - entropy(p; en)
    plogq = 0.0
    for i in eachindex(nmarg) 
        plogq += expectation((xᵢ,xᵢ₊₁)->q.f[i][xᵢ,xᵢ₊₁], nmarg[i])
    end
    plogq -= lognormalization(q)
    return plogp - plogq
end

function StatsBase.loglikelihood(chain::ChainModel{T}, x::AbstractVector{<:AbstractVector{<:Integer}}; 
        logZ = lognormalization(chain)) where T
    L = length(chain)
    all(length(xi) == L for xi in x) || throw(DimensionMismatch("inconsistent array dimensions"))
    ll = zero(T)
    for xᵃ in x
        ll += Distributions._logpdf(chain, xᵃ; logZ)
    end
    ll
end

function StatsBase.loglikelihood(chain::ChainModel, A::AbstractMatrix{<:Integer}; logZ = lognormalization(chain))
    return loglikelihood(chain, eachcol(A); logZ)
end