function _logpdf(chain::ChainModel, x; l = accumulate_left(chain.f)) 
    return log(evaluate(chain, x)) - log(normalization(chain; l))
end

# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoids creating a `Weight` object
# assumes the input vector is normalized
function sample_noalloc(rng::AbstractRNG, w) 
    t = rand(rng) * sum(w)
    i = 0
    cw = 0.0
    for p in w
        cw += p
        i += 1
        cw > t && return i
    end
    @assert false
end

function _rand!(rng::AbstractRNG, chain::ChainModel{T}, x::AbstractVector{<:Integer}) where {T<:Real}
    r = accumulate_right(chain)
    x[begin] = sample_noalloc(rng, first(r))
    for (i,q) in zip(Iterators.drop(eachindex(x), 1), Iterators.drop(nstates(chain), 1))
        p = (chain.f[i-1][x[i-1],xᵢ] * r[i+1][xᵢ] / r[i][x[i-1]] for xᵢ in 1:q)
        x[i] = sample_noalloc(rng, p)
    end
    x
end

# function _expectation(f, p::Array{<:Real, N}) where N
#     sum(f(x...) * p[x...] for x in Iterators.product(axes(p)...))
# end
# _expectation(p) = _expectation(identity, p)

expectation(f, p::Matrix{<:Real}) = sum(f(xi, xj) * p[xi, xj] for xi in axes(p,1), xj in axes(p,2))
expectation(f, p::Vector{<:Real}) = sum(f(xi) * p[xi] for xi in eachindex(p))
expectation(p) = expectation(identity, p)

function mean(chain::ChainModel; p = marginals(chain))
    return [expectation(pᵢ) for pᵢ in p]
end

function var(chain::ChainModel; p = marginals(chain))
    return [expectation(abs2, pᵢ) - expectation(pᵢ)^2 for pᵢ in p]
end

function entropy(chain::ChainModel; p = neighbor_marginals(chain))
    logZ = log(normalization(chain)) 
    avg_logf = sum(expectation((xᵢ,xᵢ₊₁)->log(chain.f[i][xᵢ,xᵢ₊₁]), p[i]) for i in eachindex(p))
    return logZ - avg_logf
end