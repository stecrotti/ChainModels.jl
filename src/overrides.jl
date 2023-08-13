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

struct ChainSampler{T,L,U} <: Sampleable{Multivariate,Discrete} where {T<:Real,L,U<:ChainModel{T,L}}
    chain :: U
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

function loglikelihood(chain::ChainModel, A::AbstractMatrix{<:Integer}; logZ = lognormalization(chain))
    size(A, 1) == length(chain) || throw(DimensionMismatch("number of rows of `A` must match size of variable, got $(size(A, 1)) and $(length(chain))."))
    # return sum(_logpdf(chain, xᵃ; logZ) for xᵃ in eachcol(A); init=0.0)
    ll = 0.0
    for xᵃ in eachcol(A)
        ll += _logpdf(chain, xᵃ; logZ)
    end
    return ll
end

function loglikelihood(chain::ChainModel, x::AbstractVector{<:AbstractVector{<:Integer}}; logZ = lognormalization(chain))
    L = length(chain)
    all(length(xi) == L for xi in x) || throw(DimensionMismatch("inconsistent array dimensions"))
    return sum(_logpdf(chain, xᵃ; logZ) for xᵃ in x)
end

# function expectation(f, p::Array{<:Real, N}) where N
#     sum(f(x...) * p[x...] for x in Iterators.product(axes(p)...))
# end
# expectation(p) = expectation(identity, p)


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

function entropy(chain::ChainModel; nmarg = neighbor_marginals(chain))
    logZ = lognormalization(chain)
    avg_logf = 0.0
    for (fᵢ,pᵢ) in zip(chain.f, nmarg)
        avg_logf += expectation((xᵢ,xᵢ₊₁)->fᵢ[xᵢ,xᵢ₊₁], pᵢ)
    end
    # avg_logf = sum(expectation((xᵢ,xᵢ₊₁)->fᵢ[xᵢ,xᵢ₊₁], pᵢ) for (fᵢ,pᵢ) in zip(chain.f, nmarg))
    return logZ - avg_logf
end

function kldivergence(p::ChainModel, q::ChainModel; nmarg = neighbor_marginals(p))
    plogp = - entropy(p; nmarg)
    plogq = 0.0
    for i in eachindex(nmarg) 
        plogq += expectation((xᵢ,xᵢ₊₁)->q.f[i][xᵢ,xᵢ₊₁], nmarg[i])
    end
    plogq -= lognormalization(q)
    # plogq = - lognormalization(q) + 
    #   sum(expectation((xᵢ,xᵢ₊₁)->q.f[i][xᵢ,xᵢ₊₁], nmarg[i]) for i in eachindex(nmarg))
    return plogp - plogq
end

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
function loglikelihood_gradient(chain::ChainModel{T}, x;
        neigmarg = neighbor_marginals(chain)) where {T}
    loglikelihood_gradient!(deepcopy(chain.f), chain, x; neigmarg)
end