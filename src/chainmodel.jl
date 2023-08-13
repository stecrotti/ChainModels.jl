struct ChainModel{T<:Real,L,BC<:BoundaryConditions} <: DiscreteMultivariateDistribution
    f :: Vector{Matrix{T}}

    function ChainModel{T,L,BC}(f::Vector{Matrix{T}}) where {T<:Real,L,BC<:BoundaryConditions}
        _checksizes(f)
        new{T,L,BC}(f)
    end
end

const OpenChainModel{T,L} = ChainModel{T,L,Open}
const PeriodicChainModel{T,L} = ChainModel{T,L,Periodic}

function ChainModel(f::Vector{Matrix{T}}, ::Type{Open}) where {T<:Real}
    _checksizes(f)
    L = length(f) + 1
    return ChainModel{T,L,Open}(f)
end
function ChainModel(f::Vector{Matrix{T}}, ::Type{Periodic}) where {T<:Real}
    _checksizes(f)
    L = length(f)
    size(f[1], 1) == size(f[end], 2) || throw(ArgumentError("Matrix sizes must be consistent"))  
    return ChainModel{T,L,Periodic}(f)
end
ChainModel(f) = ChainModel(f, Open)

function _checksizes(f::Vector{Matrix{T}}) where {T<:Real}
    all( size(f[i],2) == size(f[i+1],1) for i in 1:length(f) - 1 ) || throw(ArgumentError("Matrix sizes must be consistent"))   
    return nothing
end

accumulate_left(chain::ChainModel{T,L,BC}) where {T,L,BC} = accumulate_left(chain.f, BC)
accumulate_right(chain::ChainModel{T,L,BC}) where {T,L,BC} = accumulate_right(chain.f, BC)
accumulate_middle(chain::ChainModel{T,L,BC}) where {T,L,BC} = accumulate_middle(chain.f, BC)

length(::ChainModel{T,L}) where {T,L} = L

nstates(chain::ChainModel{T,L,BC}) where {T,L,BC} = NTuple{L,Int}(nstates(chain.f, BC))

function show(io::IO, ::ChainModel{T,L,BC}) where {T,L,BC}
    println(io, "ChainModel with $L variables, $BC boundary conditions, eltype $T")
end

function evaluate_matrices(chain::OpenChainModel, x)
    (chain.f[i][x[i],x[i+1]] for i in eachindex(chain.f))
end
function evaluate_matrices(chain::PeriodicChainModel, x)
    L = length(chain)
    (chain.f[i][x[i],x[mod1(i+1, L)]] for i in eachindex(chain.f))
end

function logevaluate(chain::ChainModel, x)
    sum(evaluate_matrices(chain, x); init=0.0)
end

function evaluate(chain::ChainModel, x)
    exp(logevaluate(chain, x)) 
end

lognormalization(chain::OpenChainModel; l = accumulate_left(chain)) = logsumexp(last(l))
normalization(chain::ChainModel; l = accumulate_left(chain)) = exp(lognormalization(chain; l))

function lognormalization(chain::PeriodicChainModel; l = accumulate_left(chain))
    qc, c = findcenter(chain.f)
    lc1 = l[mod1(c-1, length(chain))]
    return logsumexp(lc1[x] for x in diagind(lc1))
end

function normalize!(chain::ChainModel; logZ = lognormalization(chain))
    for fᵢ in chain.f
        fᵢ .-= logZ / length(chain.f)
    end
    chain
end

normalize(chain::ChainModel; logZ = lognormalization(chain)) = normalize!(deepcopy(chain); logZ)

function marginals(chain::OpenChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)) do i
        pᵢ = [l[i-1][xᵢ] + r[i+1][xᵢ] for xᵢ in eachindex(l[i-1])]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end
function marginals(chain::PeriodicChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    L = length(chain)
    return map(1:length(chain)) do i
        pᵢ = [ logsumexp(l[mod1(i-1,L)][xc,xᵢ] + r[mod1(i+1,L)][xᵢ,xc] 
            for xc in axes(l[mod1(i-1,L)], 1)) for xᵢ in axes(l[mod1(i-1,L)], 2) ]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end 

function neighbor_marginals(chain::OpenChainModel;
        l = accumulate_left(chain), r = accumulate_right(chain))
    return map(1:length(chain)-1) do i
        pᵢ = [l[i-1][xᵢ] + chain.f[i][xᵢ,xᵢ₊₁] + r[i+2][xᵢ₊₁] 
            for xᵢ in eachindex(l[i-1]), xᵢ₊₁ in eachindex(r[i+2])]
        pᵢ .-= logsumexp(pᵢ)
        pᵢ .= exp.(pᵢ)
    end
end

function pair_marginals(chain::OpenChainModel{T};
        l = accumulate_left(chain), r = accumulate_right(chain), 
        m = accumulate_middle(chain)) where T
    L = length(chain)
    p = [zeros(T, q1, q2) for q1 in nstates(chain), q2 in nstates(chain)]
    for i in 1:L-1
        for j in i+1:L
            for xᵢ in axes(p[i,j], 1)
                for xⱼ in axes(p[i,j], 2)
                    p[i,j][xᵢ,xⱼ] = l[i-1][xᵢ] + m[i,j][xᵢ,xⱼ] + r[j+1][xⱼ]
                end
            end
            p[i,j] .-= logsumexp(p[i,j])
            p[i,j] .= exp.(p[i,j])
            p[j,i] .= p[i,j]'
        end
    end
    p
end