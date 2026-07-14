function compute_empirical_Kmarginals(X, K::Integer; qs=tuple(maximum(X, dims=2)...))
    L, N = size(X)
    fK = [zeros(qs[i:i+K-1]...) for i in 1:L-K+1]
    fKm1 = [zeros(qs[i:i+K-2]...) for i in 1:L-K+2]

    for n in axes(X, 2)
        for i in 1:L-K+1
            fK[i][X[i:i+K-1, n]...] += 1/N
        end
        for i in 1:L-K+2
            fKm1[i][X[i:i+K-2, n]...] += 1/N
        end
    end

    return fK, fKm1
end


function compute_corrections(fK, fKm1, K)

    function _f_left(fK, fKm1, i)
        repeat(fKm1[i], outer=tuple(fill(1, ndims(fKm1[i]))..., size(fK[i])[end]))
    end

    function _f_right(fK, fKm1, i)
        permutedims(repeat(fKm1[i+1], outer=tuple(fill(1, ndims(fKm1[i+1]))..., size(fK[i])[1])), 
                    circshift(1:K, 1))
    end

    g = map(eachindex(fK)) do i
        if i == 1
            gi = _f_right(fK, fKm1, i)
            @assert size(gi) == size(fK[i])
            gi
        elseif i == length(fK)
            gi = _f_left(fK, fKm1, i)
            @assert size(gi) == size(fK[i])
            gi
        else
            gi = _f_left(fK, fKm1, i) .* _f_right(fK, fKm1, i)
            @assert size(gi) == size(fK[i])
            gi
        end
    end

    return g
end

function Distributions.fit_mle(::Type{KChainModel}, K::Integer, X::AbstractMatrix{<:Integer}; 
    qs=tuple(maximum(X, dims=2)...))

    fK, fKm1 = compute_empirical_Kmarginals(X, K; qs=qs)
    g = compute_corrections(fK, fKm1, K)
    f = map(zip(fK, g)) do (fKi, gi)
        fi = log.(fKi ./ sqrt.(gi))
        fi[isnan.(fi)] .= - Inf
        fi
    end

    return KChainModel(f)
end

function Distributions.fit_mle(::Type{ChainModel}, X::AbstractMatrix{<:Integer}; 
    qs=tuple(maximum(X, dims=2)...))

    return Distributions.fit_mle(KChainModel, 2, X; qs=qs)
end