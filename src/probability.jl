function _logpdf(model::ChainModel, x; l = accumulate_left(model.f)) 
    return log(evaluate(model, x)) - log(normalization(model; l))
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

function _rand!(rng::AbstractRNG, model::ChainModel{T}, x::AbstractVector{<:Integer}) where {T<:Real}
    r = accumulate_right(model)
    x[begin] = sample_noalloc(rng, first(r))
    for (i,q) in zip(Iterators.drop(eachindex(x), 1), Iterators.drop(nstates(model), 1))
        p = (model.f[i-1][x[i-1],xᵢ] * r[i+1][xᵢ] / r[i][x[i-1]] for xᵢ in 1:q)
        x[i] = sample_noalloc(rng, p)
    end
    x
end