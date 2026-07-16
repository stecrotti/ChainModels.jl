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

    # reshape matrices so that matching indices are multiplying when broadcasting
    # Ex: f1(x1, x2) * f2(x2,x3) can be done by either `_f_left(f1)` so that f1 becomes of size (1, span(x1), span(x2))
    # or equivalently by `_f_right(f2)` so that f2 becomes of size (span(x2), spam(x3),1)
    _f_left(f) = reshape(f, size(f)..., 1)
    _f_right(f) = reshape(f, 1, size(f)...)

    # computes log(f / sqrt(g * gp1)) handling 0/0 cases correctly
    function combine_fg(f, g, gp1)
        # 0/0 cases:
        # 0 / sqrt(0*0) = 1
        # 0 / sqrt(0*nonzero) = 0
        exp_out = if iszero(f) && (iszero(g) && iszero(gp1))
            one(f)
        elseif iszero(f) && (iszero(g) ⊻ iszero(gp1))
            zero(f)
        else
            f / sqrt(g * gp1)
        end
        return log(exp_out)
    end

    fK, fKm1 = empirical_frequencies
    f = map(enumerate(fK)) do (i, fKi)
        (gi, gip1) = if i == 1
            1.0, _f_right(fKm1[i+1])
        elseif i == length(fK)
            _f_left(fKm1[i]), 1.0
        else
            _f_left(fKm1[i]), _f_right(fKm1[i+1])
        end
        fi = combine_fg.(fKi, gi, gip1)
        fi
    end

    return KChainModel(f)
end

function Distributions.fit_mle(::Type{ChainModel}, X::AbstractMatrix{<:Integer}; 
    qs=tuple(maximum(X, dims=2)...))

    return Distributions.fit_mle(KChainModel, 2, X; qs=qs)
end