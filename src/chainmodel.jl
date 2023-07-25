abstract type AbstractChainModel{T<:Real} <: DiscreteMultivariateDistribution; end

struct ChainModel{T} <: AbstractChainModel{T}
    f :: Vector{Matrix{T}}

    function ChainModel(f::Vector{Matrix{T}}) where {T<:Real}
        all( all(≥(0), fᵢ) for fᵢ in f ) || throw(ArgumentError("All factors should be non-negative"))
        all( size(f[i],2) == size(f[i+1],1) for i in 1:length(f)-1 ) || throw(ArgumentError("Matrix sizes must be consistent"))
        return new{T}(f)
    end
end

accumulate_left(model::ChainModel) = accumulate_left(model.f)
accumulate_right(model::ChainModel) = accumulate_right(model.f)
accumulate_middle(model::ChainModel) = accumulate_middle(model.f)

length(model::ChainModel) = length(model.f) + 1

nstates(model::ChainModel) = nstates(model.f)

function show(io::IO, model::ChainModel{T}) where T
    L = length(model)
    println(io, "ChainModel{$T} with $L variables")
end

function evaluate(model::ChainModel, x)
    L = length(model)
    length(x) == L || throw(ArgumentError("Length of `x` should match the number of variables. Got $(length(x)) and $L."))
    prod(model.f[i][x[i],x[i+1]] for i in eachindex(model.f); init=1.0) 
end

normalization(model::ChainModel; l = accumulate_left(model.f)) = sum(last(l))

function marginals(model::ChainModel;
        l = accumulate_left(model), r = accumulate_right(model))
    return map(1:length(model)) do i
        pᵢ = l[i-1]' .* r[i+1]
        pᵢ ./= sum(pᵢ)
    end
end 

function neighbor_marginals(model::ChainModel;
        l = accumulate_left(model), r = accumulate_right(model))
    return map(1:length(model)-1) do i
        pᵢ = l[i-1]' .* model.f[i] .* r[i+2]'
        pᵢ ./= sum(pᵢ)
    end
end