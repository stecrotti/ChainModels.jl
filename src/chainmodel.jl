abstract type AbstractChainModel{T<:Real} <: DiscreteMultivariateDistribution; end

struct ChainModel{T} <: AbstractChainModel{T}
    f :: Vector{Matrix{T}}

    function ChainModel(f::Vector{Matrix{T}}) where {T<:Real}
        all( all(≥(0), fᵢ) for fᵢ in f ) || throw(ArgumentError("All factors should be non-negative"))
        all( size(f[i],2) == size(f[i+1],1) for i in 1:length(f)-1 ) || throw(ArgumentError("Matrix sizes must be consistent"))
        return new{T}(f)
    end
end

accumulate_left(chain::ChainModel) = accumulate_left(chain.f)
accumulate_right(chain::ChainModel) = accumulate_right(chain.f)
accumulate_middle(chain::ChainModel) = accumulate_middle(chain.f)

length(chain::ChainModel) = length(chain.f) + 1

nstates(chain::ChainModel) = nstates(chain.f)

function show(io::IO, chain::ChainModel{T}) where T
    L = length(chain)
    println(io, "ChainModel{$T} with $L variables")
end

function evaluate(chain::ChainModel, x)
    L = length(chain)
    length(x) == L || throw(ArgumentError("Length of `x` should match the number of variables. Got $(length(x)) and $L."))
    prod(chain.f[i][x[i],x[i+1]] for i in eachindex(chain.f); init=1.0) 
end

normalization(chain::ChainModel; l = accumulate_left(chain.f)) = sum(last(l))

function marginals(chain::ChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)) do i
        pᵢ = vec( l[i-1]' .* r[i+1] )
        pᵢ ./= sum(pᵢ)
    end
end 

function neighbor_marginals(chain::ChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)-1) do i
        pᵢ = l[i-1]' .* chain.f[i] .* r[i+2]'
        pᵢ ./= sum(pᵢ)
    end
end

function pair_marginals(chain::ChainModel{T};
        l = accumulate_left(chain), r = accumulate_right(chain), 
        m = accumulate_middle(chain)) where T
    L = length(chain)
    p = [zeros(T, q1, q2) for q1 in nstates(chain), q2 in nstates(chain)]
    for i in 1:L-1
        for j in i+1:L
            p[i,j] .= l[i-1]' .* m[i,j] .* r[j+1]'
            p[i,j] ./= sum(p[i,j])
            p[j,i] .= p[i,j]'
        end
    end
    p
end