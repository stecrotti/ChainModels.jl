"""
    compute_empirical_Kmarginals(X::AbstractMatrix{<:Integer}, K::Integer; qs=tuple(maximum(X, dims=2)...), eps=0.0)

Compute `K`- and `K-1`-body empirical frequencies from data `X` which is a matrix `L`x`N` where `L` is the chain length and `N` the number of samples.
"""
function compute_empirical_Kmarginals(X::AbstractMatrix{<:Integer}, K::Integer; 
    qs=tuple(maximum(X, dims=2)...), eps=0.0)

    L, N = size(X)
    fK = [zeros(qs[i:i+K-1]...) for i in 1:L-K+1]
    fKm1 = [zeros(qs[i:i+K-2]...) for i in 1:L-K+2]


    for i in 1:L-K+1
        for n in axes(X, 2)
            fK[i][X[i:i+K-1, n]...] += 1
        end
        clamp!(fK[i], eps, N * (1 - eps))
        fK[i] ./= sum(fK[i])
    end
    for i in 1:L-K+2
        for n in axes(X, 2)
            fKm1[i][X[i:i+K-2, n]...] += 1
        end
        clamp!(fKm1[i], eps, N * (1 - eps))
        fKm1[i] ./= sum(fKm1[i])
    end

    return fK, fKm1
end

function Distributions.fit_mle(::Type{KChainModel}, K::Integer, X::AbstractMatrix{<:Integer}; 
        qs=tuple(maximum(X, dims=2)...),
        empirical_frequencies = compute_empirical_Kmarginals(X, K; qs=qs))

    function _f_left(fK, fKm1, i)
        repeat(fKm1[i], outer=tuple(fill(1, ndims(fKm1[i]))..., size(fK[i])[end]))
    end
    function _f_right(fK, fKm1, i)
        permutedims(repeat(fKm1[i+1], outer=tuple(fill(1, ndims(fKm1[i+1]))..., size(fK[i])[1])), 
                    circshift(1:K, 1))
    end

    fK, fKm1 = empirical_frequencies
    f = map(enumerate(fK)) do (i, fKi)
        (gi, gip1) = if i == 1
            1.0, _f_right(fK, fKm1, i)
        elseif i == length(fK)
            _f_left(fK, fKm1, i), 1.0
        else
            _f_left(fK, fKm1, i), _f_right(fK, fKm1, i)
        end
        expfi = fKi ./ sqrt.(gi .* gip1)
        # 0/0 cases:
        # 0 / sqrt(0*0) = 1
        # 0 / sqrt(0*nonzero) = 0
        expfi[iszero.(fKi) .&& (iszero.(gi) .&& iszero.(gip1))] .= 1
        expfi[iszero.(fKi) .&& (iszero.(gi) .⊻ iszero.(gip1))] .= 0
        fi = log.(expfi)
        fi[isnan.(fi)] .= - Inf
        fi
    end

    return KChainModel(f)
end

function Distributions.fit_mle(::Type{ChainModel}, X::AbstractMatrix{<:Integer}; 
    qs=tuple(maximum(X, dims=2)...))

    return Distributions.fit_mle(KChainModel, 2, X; qs=qs)
end