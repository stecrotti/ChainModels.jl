# number of values each variable in the chain can take
function nstates(f::Vector{Matrix{T}}) where {T<:Real}
    N = length(f)
    Tuple(i == N + 1 ? size(f[end],2) : size(f[i],1) for i in 1:N+1)
end

function accumulate_left!(l, f::Vector{Matrix{T}}) where {T<:Real}
    l[0] .= 0
    for i in eachindex(f)
        for xᵢ₊₁ in eachindex(l[i])
            l[i][xᵢ₊₁] = logsumexp(l[i-1][xᵢ] + f[i][xᵢ,xᵢ₊₁] for xᵢ in eachindex(l[i-1]))
        end
    end
    l
end

function accumulate_left(f::Vector{Matrix{T}}) where {T<:Real}
    l = OffsetArray([zeros(T, 1, q) for q in nstates(f)], -1)
    accumulate_left!(l, f)
end

function accumulate_right!(r, f::Vector{Matrix{T}}) where {T<:Real}
    r[end] .= 0
    for i in reverse(eachindex(f))
        for xᵢ in eachindex(r[i+1])
            r[i+1][xᵢ] = logsumexp(f[i][xᵢ,xᵢ₊₁] + r[i+2][xᵢ₊₁] for xᵢ₊₁ in eachindex(r[i+2]))
        end
    end
    r
end

function accumulate_right(f::Vector{Matrix{T}}) where {T<:Real}
    r = OffsetArray([zeros(T, q, 1) for q in nstates(f)], +1)
    accumulate_right!(r, f)
end

function accumulate_middle!(m, f::Vector{Matrix{T}}) where {T<:Real}
    for i in eachindex(f)
        m[i,i+1] .= f[i]
    end
    for j in Iterators.drop(axes(m,2), 1)
        for i in reverse(1:j-2)
            for xᵢ in axes(m[i,j], 1)
                for xⱼ in axes(m[i,j], 2)
                    m[i, j][xᵢ,xⱼ] = logsumexp(f[i][xᵢ,xᵢ₊₁] + m[i+1,j][xᵢ₊₁,xⱼ] 
                                                        for xᵢ₊₁ in axes(m[i+1,j], 1))
                end
            end
        end
    end
    m
end

function accumulate_middle(f::Vector{Matrix{T}}) where {T<:Real}
    m = OffsetArray([zeros(T, q1, q2) for q1 in nstates(f)[1:end-1], q2 in nstates(f)[2:end]], 0, +1) 
    accumulate_middle!(m, f)
end